using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Web.Http;
using LifeinnovirorMentalHealthConsultency.Context;
using LifeinnovirorMentalHealthConsultency.Context.Tables;

namespace LifeinnovirorMentalHealthConsultency.Controllers.AdminControllers
{
    [Authorize(Roles = "Admin")]
    public class AdminSpecializationManagementController : ApiController
    {
        private LifeinnovirorContext db;    // Creating private db object to manupulate data
        public AdminSpecializationManagementController()
        {
            db = new LifeinnovirorContext(); // Initializing the database in constructor 
        }


        // it receives specialization data and add to database
        [HttpPost]
        [Route("api/admin/addSpecialization")]
        public IHttpActionResult AddSpecialization(Specialization data)
        {
            //Try-catch block to handle unintended errors
            try
            {
                // it checks if the data is valid or not
                if (ModelState.IsValid)
                {
                    // Check if specialization already exists
                    var exists = db.Specializations
                                   .Any(s => s.Name.ToLower().Trim() == data.Name.ToLower().Trim());

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
                    db.SaveChanges();

                    // Log: Successful addition
                    db.SystemLogs.Add(new SystemLog
                    {
                        ActorType = "Admin",
                        ActorId = CustomFunctions.GetAdminUserIdFromToken(User), 
                        Action = "Add Specialization",
                        Details = $"Added specialization '{data.Name}' successfully.",
                        CreatedAt = DateTime.Now
                    });
                    db.SaveChanges();

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
                return InternalServerError(new Exception("An error occurred while adding specialization: " + ex.Message));
            }
        }


        //This is retrive all the specialization data available in the database and send it
        [HttpGet]
        [Route("api/admin/getAllSpecializations")]
        public IHttpActionResult GetAllSpecializations()
        {
            //Try-catch block to handle unintended errors
            try
            {
                // Getting specialization list from database 
                var specializations = db.Specializations.ToList();

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
                return InternalServerError(new Exception("An error occurred while retrieving specializations: " + ex.Message));
            }
        }




        // This will take the new specialization data and update that specialization
        [HttpPost]
        [Route("api/admin/updateSpecialization")]
        public IHttpActionResult UpdateSpecialization(Specialization updatedData)
        {
            //Try-catch block to handle unintended errors
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
                var existingSpecialization = db.Specializations.Find(updatedData.SpecializationId);
                if (existingSpecialization == null)
                {
                    return NotFound(); // 404 if not found
                }

                // Check for duplication by name (excluding current record)
                bool isDuplicate = db.Specializations
                                     .Any(s => s.SpecializationId != updatedData.SpecializationId &&
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
                db.SaveChanges();

                // Log: Successful Update of specialization
                db.SystemLogs.Add(new SystemLog
                {
                    ActorType = "Admin",
                    ActorId = CustomFunctions.GetAdminUserIdFromToken(User),
                    Action = "Update Specialization",
                    Details = $"Updated specialization '{updatedData.Name}' successfully.",
                    CreatedAt = DateTime.Now
                });
                db.SaveChanges();

                return Ok(new
                {
                    success = true,
                    message = "Specialization updated successfully.",
                    data = existingSpecialization
                });
            }
            catch (Exception ex)
            {
                return InternalServerError(new Exception("An error occurred while Updating specializations: " + ex.Message));
            }
        }


        [HttpPost]
        [Route("api/admin/deleteSpecialization/{id}")]
        public IHttpActionResult DeleteSpecialization(int id)
        {
            try
            {
                // Find the specialization
                var specialization = db.Specializations.FirstOrDefault(s => s.SpecializationId == id);

                if (specialization == null)
                {
                    return NotFound(); // Return 404 if not found
                }

                // Find all sub-specializations associated with this specialization
                var subSpecializations = db.SubSpecializations
                                           .Where(ss => ss.SpecializationId == id)
                                           .ToList();

                // Delete all associated sub-specializations
                db.SubSpecializations.RemoveRange(subSpecializations);

                // Delete the specialization itself
                db.Specializations.Remove(specialization);

                // Save all changes to the database
                db.SaveChanges();

                // Log successful deletion 
                db.SystemLogs.Add(new SystemLog
                {
                    ActorType = "Admin",
                    ActorId = CustomFunctions.GetAdminUserIdFromToken(User),
                    Action = "Delete Specialization",
                    Details = $"Deleted specialization '{specialization.Name}' and its {subSpecializations.Count} sub-specializations.",
                    CreatedAt = DateTime.Now
                });

                db.SaveChanges();

                return Ok(new
                {
                    success = true,
                    message = "Specialization and all related sub-specializations deleted successfully.",
                    data = specialization
                });
            }
            catch (Exception ex)
            {
                return InternalServerError(new Exception("An error occurred while deleting the specialization. Details: " + ex.Message));
            }
        }



    }
}
