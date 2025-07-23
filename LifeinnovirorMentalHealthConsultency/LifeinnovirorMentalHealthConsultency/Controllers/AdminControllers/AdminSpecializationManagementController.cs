using System;
using System.Collections.Generic;
using System.Data.Entity;
using System.Linq;
using System.Net;
using System.Threading.Tasks;
using System.Web.Http;
using LifeinnovirorMentalHealthConsultency.Context;
using LifeinnovirorMentalHealthConsultency.Context.Tables;

namespace LifeinnovirorMentalHealthConsultency.Controllers.AdminControllers
{
    [Authorize(Roles = "Admin")]
    public class AdminSpecializationManagementController : ApiController
    {
        private readonly LifeinnovirorContext db;  
        public AdminSpecializationManagementController()
        {
            db = new LifeinnovirorContext(); // Initializing the database 
        }


        // it receives specialization data and add to database
        [HttpPost]
        [Route("api/admin/addSpecialization")]
        public async Task<IHttpActionResult> AddSpecialization(Specialization data)
        {
            try
            {
                // it checks if the data is valid or not
                if (ModelState.IsValid)
                {
                    // Check if specialization already exists
                    var exists = await db.Specializations
                                      .AnyAsync(s => s.Name.ToLower().Trim() == data.Name.ToLower().Trim());

                    if (exists)
                    {
                        return Content(HttpStatusCode.Conflict, new
                        {
                            success = false,
                            message = "This specialization already exists.",
                            data = data
                        });
                    }

                    // Add specialization to DB
                    db.Specializations.Add(data);

                    // Log: Successful addition
                    db.SystemLogs.Add(new SystemLog
                    {
                        ActorType = "Admin",
                        ActorId = CustomFunctions.GetAdminUserIdFromToken(User), 
                        Action = "Add Specialization",
                        Details = $"Added specialization '{data.Name}' successfully.",
                        CreatedAt = DateTime.Now
                    });
                    await db.SaveChangesAsync();

                    return Ok(new
                    {
                        success = true,
                        message = "Specialization added successfully.",
                        data = data
                    });
                }

                // if invalid data then Collect validation errors
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
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "An error occurred while adding specialization",
                    error = ex.Message
                });
            }
        }


        //This is retrive all the specialization data available in the database and send it
        [HttpGet]
        [Route("api/admin/getAllSpecializations")]
        public async Task<IHttpActionResult> GetAllSpecializations()
        {
            try
            {
                // Getting specialization list from database 
                var specializations = await db.Specializations.ToListAsync();

                // If no specialization found then it will send success message with the message
                if (specializations == null || !specializations.Any())
                {
                    return Ok(new
                    {
                        success = true,
                        message = "No specializations found.",
                        data = new List<object>()
                    });
                }

                // If specialization found then it will send all secialization objects as list
                return Ok(new
                {
                    success = true,
                    message = "Specializations retrieved successfully.",
                    data = specializations
                });
            }
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "An error occurred while retriving specialization",
                    error = ex.Message
                });
            }
        }




        // This will take the new specialization data and update that specialization
        [HttpPost]
        [Route("api/admin/updateSpecialization")]
        public async Task<IHttpActionResult> UpdateSpecialization(Specialization updatedData)
        {
            try
            {
                // If received invalid data then send the ModelState error
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

                // Searching if the specialization existed in the database or not
                var existingSpecialization = await db.Specializations.FindAsync(updatedData.SpecializationId);
                if (existingSpecialization == null)
                {
                    return NotFound(); // 404 if not found
                }

                // Check for duplication by name (excluding current record)
                bool isDuplicate =await  db.Specializations
                                         .AnyAsync(s => s.SpecializationId != updatedData.SpecializationId &&
                                          s.Name.ToLower().Trim() == updatedData.Name.ToLower().Trim());

                if (isDuplicate)
                {
                    return Content(HttpStatusCode.Conflict, new
                    {
                        success = false,
                        message = "Another specialization with this name already exists.",
                        data = updatedData
                    });
                }

                // Update fields
                existingSpecialization.Name = updatedData.Name;

                // Log: Successful Update of specialization
                db.SystemLogs.Add(new SystemLog
                {
                    ActorType = "Admin",
                    ActorId = CustomFunctions.GetAdminUserIdFromToken(User),
                    Action = "Update Specialization",
                    Details = $"Updated specialization '{updatedData.Name}' successfully.",
                    CreatedAt = DateTime.Now
                });
                await db.SaveChangesAsync();

                return Ok(new
                {
                    success = true,
                    message = "Specialization updated successfully.",
                    data = existingSpecialization
                });
            }
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "An error occurred while updating specialization",
                    error = ex.Message
                });
            }
        }


        [HttpPost]
        [Route("api/admin/deleteSpecialization/{id}")]
        public async Task<IHttpActionResult> DeleteSpecialization(int id)
        {
            try
            {
                var specialization = await db.Specializations.FirstOrDefaultAsync(s => s.SpecializationId == id);

                if (specialization == null)
                {
                    return NotFound(); // Return 404 if not found
                }

                // Find all sub-specializations associated with this specialization
                var subSpecializations = await db.SubSpecializations
                                           .Where(ss => ss.SpecializationId == id)
                                           .ToListAsync();

                // Delete all associated sub-specializations
                db.SubSpecializations.RemoveRange(subSpecializations);

                // Delete the specialization itself
                db.Specializations.Remove(specialization);

                // Log successful deletion 
                db.SystemLogs.Add(new SystemLog
                {
                    ActorType = "Admin",
                    ActorId = CustomFunctions.GetAdminUserIdFromToken(User),
                    Action = "Delete Specialization",
                    Details = $"Deleted specialization '{specialization.Name}' and its {subSpecializations.Count} sub-specializations.",
                    CreatedAt = DateTime.Now
                });

                await db.SaveChangesAsync();

                return Ok(new
                {
                    success = true,
                    message = "Specialization and all related sub-specializations deleted successfully.",
                    data = specialization
                });
            }
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "An error occurred while deleting specialization",
                    error = ex.Message
                });
            }
        }



    }
}
