using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Security.Cryptography;
using System.Text;
using System.Web.Http;
using LifeinnovirorMentalHealthConsultency.Authorization;
using LifeinnovirorMentalHealthConsultency.Context;
using LifeinnovirorMentalHealthConsultency.Context.Tables;

namespace LifeinnovirorMentalHealthConsultency.Controllers.AdminControllers
{
    [Authorize(Roles = "Admin")]
    public class AdminCrudOperationController : ApiController
    {
        private LifeinnovirorContext db;    // Creating private db object to manupulate data
        public AdminCrudOperationController()
        {
            db = new LifeinnovirorContext(); // Initializing the database in constructor 
        }


        //this will take admin data and create an admin
        [HttpPost]
        [Route("api/admin/createAdmin")]
        public IHttpActionResult CreateAdmin(Admin data)
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
                var existingEmail = db.Admins.Any(a => a.Email == data.Email);
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
                data.PasswordHash = Hashing.CreateMD5(data.PasswordHash); //creating MD5 hashing
                db.Admins.Add(data);
                db.SaveChanges();

                return Ok(new
                {
                    success = true,
                    message = "Admin created successfully.",
                    data = data,
                });
            }
            catch (Exception ex)
            {
                return InternalServerError(new Exception("An error occurred while creating admin: " + ex.Message));
            }
        }



        // this will get al the admin from database and return
        [HttpGet]
        [Route("api/admin/getAllAdmins")]
        public IHttpActionResult GetAllAdmins()
        {
            try
            {
                //getting all admin from database and making lists
                var admins = db.Admins.ToList();

                return Ok(new
                {
                    success = true,
                    message = "Admins retrieved successfully.",
                    data = admins
                });
            }
            catch (Exception ex)
            {
                return InternalServerError(new Exception("An error occurred while retriving admin: " + ex.Message));
            }
        }



        // this will take update admin data and replace with existing admin to update its data
        [HttpPost]
        [Route("api/admin/updateAdmin")]
        public IHttpActionResult UpdateAdmin(Admin updatedData)
        {
            try
            {
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
                var existingAdmin = db.Admins.Find(updatedData.AdminId);
                if (existingAdmin == null)
                {
                    return NotFound();   // if not found then send 404 status code
                }

                // checking if updated email matched with other mail or not
                var emailExists = db.Admins.Any(a => a.Email == updatedData.Email && a.AdminId != updatedData.AdminId);
                if (emailExists)
                {
                    return Ok(new
                    {
                        success = false,
                        message = "Another admin already exists with this email.",
                        data = updatedData
                    });
                }

                // cehcking if password is changed or not
                if (existingAdmin.PasswordHash != updatedData.PasswordHash)
                {
                    //if changed then hash the new password and update it
                    existingAdmin.PasswordHash = Hashing.CreateMD5(updatedData.PasswordHash);
                }
                // Update others fields also
                existingAdmin.Email = updatedData.Email;
                existingAdmin.FullName = updatedData.FullName;
                db.SaveChanges();

                return Ok(new
                {
                    success = true,
                    message = "Admin updated successfully.",
                    data = existingAdmin
                });
            }
            catch (Exception ex)
            {
                return InternalServerError(new Exception("An error occurred while updating admin: " + ex.Message));
            }
        }



        //This will delete admin by its id
        [HttpPost]
        [Route("api/admin/deleteAdmin/{id}")]
        public IHttpActionResult DeleteAdmin(int id)
        {
            try
            {
                // checking if admin exists or not
                var admin = db.Admins.Find(id);
                if (admin == null)
                {
                    return NotFound();
                }

                db.Admins.Remove(admin);  // if existes then remove
                db.SaveChanges();

                return Ok(new
                {
                    success = true,
                    message = "Admin deleted successfully.",
                    data = admin
                });
            }
            catch (Exception ex)
            {
                return InternalServerError(new Exception("An error occurred while deleting admin: " + ex.Message));
            }
        }





    }
}
