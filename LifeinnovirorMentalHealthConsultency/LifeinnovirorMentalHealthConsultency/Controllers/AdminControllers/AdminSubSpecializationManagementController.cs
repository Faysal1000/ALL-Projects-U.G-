using System;
using System.Collections.Generic;
using System.Data.Entity;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Threading.Tasks;
using System.Web.Http;
using LifeinnovirorMentalHealthConsultency.Context;
using LifeinnovirorMentalHealthConsultency.Context.Tables;

namespace LifeinnovirorMentalHealthConsultency.Controllers.AdminControllers
{
    public class AdminSubSpecializationManagementController : ApiController
    {
        private readonly LifeinnovirorContext db;  
        public AdminSubSpecializationManagementController()
        {
            db = new LifeinnovirorContext(); // Initializing the database 
        }



        // Add a new SubSpecialization 
        [HttpPost]
        [Route("api/admin/addSubSpecialization")]
        public async Task<IHttpActionResult> AddSubSpecialization(SubSpecialization data)
        {
            try
            {
                // if it is valid data
                if (ModelState.IsValid)
                {
                    // checking if the subspecialization already exists under the specialization
                    var exists = await db.SubSpecializations
                                     .AnyAsync(s => s.SpecializationId == data.SpecializationId &&
                                              s.Name.ToLower().Trim() == data.Name.ToLower().Trim());

                    if (exists)
                    {
                        return Content(HttpStatusCode.Conflict, new   //409 conflict code
                        {
                            success = false,
                            message = "This sub-specialization already exists under the selected specialization.",
                            data
                        });
                    }

                    db.SubSpecializations.Add(data);      //add data to the database

                    // Log: Successful addition
                    db.SystemLogs.Add(new SystemLog
                    {
                        ActorType = "Admin",
                        ActorId = CustomFunctions.GetAdminUserIdFromToken(User),
                        Action = "Add Sub Specialization",
                        Details = $"Added sub-specialization '{data.Name}' successfully.",
                        CreatedAt = DateTime.Now
                    });
                    await db.SaveChangesAsync();


                    return Ok(new
                    {
                        success = true,
                        message = "Sub-specialization added successfully.",
                        data
                    });
                }

                // If it is invalid data then send invalid message
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
                    errors,
                    data
                });
            }
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "An error occurred while adding sub-specialization",
                    error = ex.Message
                });
            }
        }




        // Get all SubSpecializations
        [HttpGet]
        [Route("api/admin/getAllSubSpecializations")]
        public async Task<IHttpActionResult> GetAllSubSpecializations()
        {
            try
            {
                //Only select respective fields to send. No need to send specialization also
                var subs = await db.SubSpecializations
                             .Select(s => new
                             {
                                 s.SubSpecializationId,
                                 s.Name,
                                 s.SpecializationId,
                                 SpecializationName = s.Specialization.Name
                             })
                             .ToListAsync();

                // if there is no subspecialization
                if (subs == null || !subs.Any())
                {
                    return Ok(new
                    {
                        success = true,
                        message = "No sub-specializations found.",
                        data = new List<object>()
                    });
                }

                //send subspecialization details
                return Ok(new
                {
                    success = true,
                    message = "Sub-specializations retrieved successfully.",
                    data = subs
                });
            }
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "An error occurred while retriving all sub-specialization",
                    error = ex.Message
                });
            }
        }



        // Get SubSpecializations by its id
        [HttpGet]
        [Route("api/admin/getSubSpecialization/{id}")]
        public async Task<IHttpActionResult> GetSubSpecialization(int id)
        {
            try
            {
                var subs = await db.SubSpecializations
                   .Where(s => s.SubSpecializationId == id)
                   .Select(s => new
                   {
                       s.SubSpecializationId,
                       s.Name,
                       s.SpecializationId,
                       SpecializationName = s.Specialization.Name
                   })
                   .FirstOrDefaultAsync();

                // if there is no subspecialization
                if (subs == null)
                {
                    return NotFound();
                }

                //send subspecialization details
                return Ok(new
                {
                    success = true,
                    message = "Sub-specialization retrieved successfully.",
                    data = subs
                });
            }
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "An error occurred while retriving sub-specialization with its id",
                    error = ex.Message
                });
            }
        }



        // This will take specialization id and give all the associated subspecial list
        [HttpGet]
        [Route("api/admin/getSubSpecializationsBySpecialization/{id}")]
        public async Task<IHttpActionResult> GetSubSpecializationsBySpecialization(int id)
        {
            try
            {
                // get all subspecialization where it matches given specialization id
                var subSpecializations =await db.SubSpecializations
                                           .Where(ss => ss.SpecializationId == id)
                                           .Select(s => new   // only select necessary data
                                           {
                                               s.SubSpecializationId,
                                               s.Name,
                                               s.SpecializationId,
                                               SpecializationName = s.Specialization.Name
                                           })
                                          .ToListAsync();

                //if there is no subspecialization found under that specialization
                if (subSpecializations == null || !subSpecializations.Any())
                {
                    return Ok(new
                    {
                        success = true,
                        message = "No sub-specializations found for the given Specialization ID.",
                        data = new List<object>()
                    });
                }

                //send subspecialization data
                return Ok(new
                {
                    success = true,
                    message = "Sub-specializations retrieved successfully.",
                    data = subSpecializations
                });
            }
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "An error occurred while retriving sub-specialization with specialization id",
                    error = ex.Message
                });
            }
        }



        // Update an existing SubSpecialization
        [HttpPost]
        [Route("api/admin/updateSubSpecialization")]
        public async Task<IHttpActionResult> UpdateSubSpecialization(SubSpecialization updatedData)
        {
            try
            {
                // if ivalid data then send the error message
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
                        errors,
                        data = updatedData
                    });
                }

                // Checking if the subspecialization is exists or not
                var existing = await db.SubSpecializations.FindAsync(updatedData.SubSpecializationId);
                if (existing == null)
                {
                    return NotFound();  //404 status code
                }

                //checking if duplicate subspecialization given or not
                // condition: ignoring same subspecialization, and searching in same specialization
                bool isDuplicate =await db.SubSpecializations
                                     .AnyAsync(s => s.SubSpecializationId != updatedData.SubSpecializationId &&
                                               s.Name.ToLower().Trim() == updatedData.Name.ToLower().Trim() &&
                                               s.SpecializationId == updatedData.SpecializationId);

                // if duplicate name entered then send conflict status code
                if (isDuplicate)
                {
                    return Content(HttpStatusCode.Conflict, new
                    {
                        success = false,
                        message = "Another sub-specialization with the same name exists under this specialization.",
                        data = updatedData
                    });
                }

                // if all success then update the subspecialization name
                existing.Name = updatedData.Name;

                // Log: Successful update
                db.SystemLogs.Add(new SystemLog
                {
                    ActorType = "Admin",
                    ActorId = CustomFunctions.GetAdminUserIdFromToken(User),
                    Action = "Update Sub Specialization",
                    Details = $"Updated sub-specialization '{updatedData.Name}' successfully.",
                    CreatedAt = DateTime.Now
                });
                await db.SaveChangesAsync();

                return Ok(new
                {
                    success = true,
                    message = "Sub-specialization updated successfully.",
                    data = existing
                });
            }
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "An error occurred while updating sub-specialization",
                    error = ex.Message
                });
            }
        }




        // Delete SubSpecialization by ID
        [HttpPost]
        [Route("api/admin/deleteSubSpecialization/{id}")]
        public async Task<IHttpActionResult> DeleteSubSpecialization(int id)
        {
            try
            {
                // finding if the subspecialization exists or not
                var sub = await db.SubSpecializations.FirstOrDefaultAsync(s => s.SubSpecializationId == id);

                if (sub == null)
                {
                    return NotFound();   // if not found then send 404 status code
                }

                db.SubSpecializations.Remove(sub);  // if found then delete that

                // Log: Successful deletion
                db.SystemLogs.Add(new SystemLog
                {
                    ActorType = "Admin",
                    ActorId = CustomFunctions.GetAdminUserIdFromToken(User),
                    Action = "Delete Sub Specialization",
                    Details = $"Deleted sub-specialization '{sub.Name}' successfully.",
                    CreatedAt = DateTime.Now
                });
                await db.SaveChangesAsync();

                return Ok(new
                {
                    success = true,
                    message = "Sub-specialization deleted successfully.",
                    data = sub
                });
            }
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "An error occurred while deleting sub-specialization",
                    error = ex.Message
                });
            }
        }


    }
}
