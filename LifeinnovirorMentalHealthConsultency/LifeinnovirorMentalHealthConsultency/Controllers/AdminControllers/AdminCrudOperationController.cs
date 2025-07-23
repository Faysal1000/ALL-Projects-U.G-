using System;
using System.Collections.Generic;
using System.Data.Entity;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using System.Web.Http;
using LifeinnovirorMentalHealthConsultency.Authorization;
using LifeinnovirorMentalHealthConsultency.Context;
using LifeinnovirorMentalHealthConsultency.Context.Tables;
using LifeinnovirorMentalHealthConsultency.Models;

namespace LifeinnovirorMentalHealthConsultency.Controllers.AdminControllers
{
    [Authorize(Roles = "Admin")]
    public class AdminCrudOperationController : ApiController
    {
        private readonly LifeinnovirorContext db;    
        public AdminCrudOperationController()
        {
            db = new LifeinnovirorContext(); // Initializing the database 
        }


        //this will take admin data and create an admin
        [HttpPost]
        [Route("api/admin/createAdminAccount")]
        public async Task<IHttpActionResult> CreateAdmin(Admin data)
        {
            try
            {
                // if the data is invalid then send error messages
                if (!ModelState.IsValid)
                {
                    var errors = ModelState.Where(ms => ms.Value.Errors.Count > 0)
                                           .Select(ms => new
                                           {
                                               Field = ms.Key,
                                               Errors = ms.Value.Errors.Select(e => e.ErrorMessage).ToList()
                                           });

                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = "Validation failed.",
                        errors = errors,
                        data = data
                    });

                }

                // check if email already exists or not
                var existingEmail = await db.Admins.AnyAsync(a => a.Email == data.Email);
                if (existingEmail)
                {
                    return Ok(new
                    {
                        success = false,
                        message = "Admin with this email already exists.",
                        data = data
                    });
                }

                //if all successfull then secure the password by hashing
                data.PasswordHash = CustomFunctions.GetSha256HashBase64(data.PasswordHash); 
                db.Admins.Add(data);

                // Log: Successful creation
                db.SystemLogs.Add(new SystemLog
                {
                    ActorType = "Admin",
                    ActorId = CustomFunctions.GetAdminUserIdFromToken(User),
                    Action = "Create Admin",
                    Details = $"Created admin '{data.Email}' successfully.",
                    CreatedAt = DateTime.Now
                });
                await db.SaveChangesAsync();

                return Ok(new
                {
                    success = true,
                    message = "Admin created successfully.",
                    data = data,
                });
            }
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "Unexpected error while creating admin account.",
                    error = ex.Message
                });
            }
        }



        // this will get al the admin from database and return
        [HttpGet]
        [Route("api/admin/getAllAdminAccounts")]
        public async Task<IHttpActionResult> GetAllAdmins()
        {
            try
            {
                //getting all admin from database and making lists
                var admins = await db.Admins.ToListAsync();

                return Ok(new
                {
                    success = true,
                    message = "Admins retrieved successfully.",
                    data = admins
                });
            }
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "Unexpected error while retriving admins",
                    error = ex.Message
                });
            }
        }



        // this will get current logged admin from database and return
        [HttpGet]
        [Route("api/admin/getAdminAccount")]
        public async Task<IHttpActionResult> GetAdminAccount()
        {
            try
            {
                //getting admin from database
                var admin = await db.Admins.FindAsync(CustomFunctions.GetAdminUserIdFromToken(User));

                return Ok(new
                {
                    success = true,
                    message = "Admin retrieved successfully.",
                    data = admin
                });
            }
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "Unexpected error while retriving admin",
                    error = ex.Message
                });
            }
        }


        // this will get accociated id admin from database and return
        [HttpGet]
        [Route("api/admin/getAdminAccount/{id}")]
        public async Task<IHttpActionResult> GetAdminAccount(int id)
        {
            try
            {
                //getting admin from database
                var admin = await db.Admins.FindAsync(id);

                return Ok(new
                {
                    success = true,
                    message = "Admin retrieved successfully.",
                    data = admin
                });
            }
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "Unexpected error while retriving admin",
                    error = ex.Message
                });
            }
        }




        // this will take update admin data and replace with existing admin to update its data
        [HttpPut]
        [Route("api/admin/updateAdminAccount")]
        public async Task<IHttpActionResult> UpdateAdmin(Admin updatedData)
        {
            try
            {
                // Remove password validation manually before checking model state
                ModelState.Remove("PasswordHash");

                // if the data is invalid then send error message
                if (!ModelState.IsValid)
                {
                    var errors = ModelState.Where(ms => ms.Value.Errors.Count > 0)
                                           .Select(ms => new
                                           {
                                               Field = ms.Key,
                                               Errors = ms.Value.Errors.Select(e => e.ErrorMessage).ToList()
                                           });

                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = "Validation failed.",
                        errors = errors,
                        data = updatedData
                    });
                }

                // checking if the admin exists in database or not
                var existingAdmin = await db.Admins.FindAsync(updatedData.AdminId);
                if (existingAdmin == null)
                {
                    return NotFound();   
                }

                //checking if associated admin changing its data or not
                if (CustomFunctions.GetAdminUserIdFromToken(User) != updatedData.AdminId)
                {
                    return Content(HttpStatusCode.Forbidden, new
                    {
                        success = false,
                        message = "Unauthorized data manupulation.",
                    });
                }


                // checking if updated email matched with other mail or not
                var emailExists = await db.Admins.AnyAsync(a => a.Email == updatedData.Email && 
                                                           a.AdminId != updatedData.AdminId);
                if (emailExists)
                {
                    return Ok(new
                    {
                        success = false,
                        message = "Another admin already exists with this email.",
                        data = updatedData
                    });
                }


                // Update others fields also
                existingAdmin.Email = updatedData.Email;
                existingAdmin.FullName = updatedData.FullName;

                // Log: Successful update
                db.SystemLogs.Add(new SystemLog
                {
                    ActorType = "Admin",
                    ActorId = updatedData.AdminId,
                    Action = "Update Admin",
                    Details = $"Updated admin '{updatedData.Email}' successfully.",
                    CreatedAt = DateTime.Now
                });
                await db.SaveChangesAsync();


                return Ok(new
                {
                    success = true,
                    message = "Admin updated successfully.",
                    data = existingAdmin
                });
            }
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "Unexpected error while updating admin",
                    error = ex.Message
                });
            }
        }


        //This will delete current logged admin account
        [HttpDelete]
        [Route("api/admin/deleteAdminAccount/{id}")]
        public async Task<IHttpActionResult> DeleteAdmin()
        {
            try
            {
                // checking if admin exists or not
                var admin = await db.Admins.FindAsync(CustomFunctions.GetAdminUserIdFromToken(User));
                if (admin == null)
                {
                    return NotFound();
                }

                db.Admins.Remove(admin);  // if existes then remove

                // Log: Successful deletion
                db.SystemLogs.Add(new SystemLog
                {
                    ActorType = "Admin",
                    ActorId = CustomFunctions.GetAdminUserIdFromToken(User),
                    Action = "Delete Admin",
                    Details = $"Deleted admin '{admin.Email}', 'Id = {admin.AdminId}' successfully.",
                    CreatedAt = DateTime.Now
                });
                await db.SaveChangesAsync();

                return Ok(new
                {
                    success = true,
                    message = "Admin deleted successfully.",
                    data = admin
                });
            }
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "Unexpected error while updating admin",
                    error = ex.Message
                });
            }
        }


        //This will delete admin by its id
        [HttpDelete]
        [Route("api/admin/deleteAdminAccount/{id}")]
        public async Task<IHttpActionResult> DeleteAdmin(int id)
        {
            try
            {
                // checking if admin exists or not
                var admin = await db.Admins.FindAsync(id);
                if (admin == null)
                {
                    return NotFound();
                }

                db.Admins.Remove(admin);  // if existes then remove

                // Log: Successful deletion
                db.SystemLogs.Add(new SystemLog
                {
                    ActorType = "Admin",
                    ActorId = CustomFunctions.GetAdminUserIdFromToken(User),
                    Action = "Delete Admin",
                    Details = $"Deleted admin '{admin.Email}', 'Id = {admin.AdminId}' successfully.",
                    CreatedAt = DateTime.Now
                });
                await db.SaveChangesAsync();

                return Ok(new
                {
                    success = true,
                    message = "Admin deleted successfully.",
                    data = admin
                });
            }
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "Unexpected error while updating admin",
                    error = ex.Message
                });
            }
        }



        [HttpPost]
        [Route("api/admin/changePassword")]
        public async Task<IHttpActionResult> ChangePassword(ChangePasswordModel model)
        {
            try
            {
                // Model validation
                if (!ModelState.IsValid)
                {
                    var errors = ModelState.Where(ms => ms.Value.Errors.Any())
                        .Select(ms => new
                        {
                            Field = ms.Key,
                            Errors = ms.Value.Errors.Select(e => e.ErrorMessage).ToList()
                        });

                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = "Validation failed.",
                        errors = errors
                    });
                }


                // Verify if current user exists or not 
                var admin = await db.Admins.FindAsync(CustomFunctions.GetAdminUserIdFromToken(User));
                if (admin == null)
                {
                    return Content(HttpStatusCode.NotFound, new
                    {
                        success = false,
                        message = "Admin not found."
                    });
                }

                //verify current password
                if (admin.PasswordHash != CustomFunctions.GetSha256HashBase64(model.CurrentPassword))
                {
                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = "Current password is incorrect."
                    });
                }

                // Update password
                admin.PasswordHash = CustomFunctions.GetSha256HashBase64(model.NewPassword);

                // Log addition
                db.SystemLogs.Add(new SystemLog
                {
                    ActorType = "Admin",
                    ActorId = admin.AdminId,
                    Action = "Change Password",
                    Details = $"Doctor '{admin.Email}' changed their account password.",
                    CreatedAt = DateTime.Now
                });
                await db.SaveChangesAsync();

                return Ok(new
                {
                    success = true,
                    message = "Password changed successfully."
                });
            }
            catch (Exception ex)
            {
                // Unexpected server error
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "An unexpected error occurred while changing password.",
                    error = ex.Message
                });
            }

        }



    }
}
