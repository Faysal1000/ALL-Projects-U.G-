using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Web.Http;
using LifeinnovirorMentalHealthConsultency.Context;
using LifeinnovirorMentalHealthConsultency.Context.Tables;

namespace LifeinnovirorMentalHealthConsultency.Controllers.AdminControllers
{
    public class AdminSubSpecializationManagementController : ApiController
    {
        private LifeinnovirorContext db;    // Creating private db object to manupulate data
        public AdminSubSpecializationManagementController()
        {
            db = new LifeinnovirorContext(); // Initializing the database in constructor 
        }



        // Add a new SubSpecialization 
        [HttpPost]
        [Route("api/admin/addSubSpecialization")]
        public IHttpActionResult AddSubSpecialization(SubSpecialization data)
        {
            //Try-catch block to handle unintended errors
            try
            {
                // if it is valid data
                if (ModelState.IsValid)
                {
                    // checking if the subspecialization already exists under the specialization
                    var exists = db.SubSpecializations
                                   .Any(s => s.SpecializationId == data.SpecializationId &&
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
                    db.SaveChanges();

                    // Log: Successful addition
                    db.SystemLogs.Add(new SystemLog
                    {
                        ActorType = "Admin",
                        ActorId = CustomFunctions.GetAdminUserIdFromToken(User),
                        Action = "Add Sub Specialization",
                        Details = $"Added sub-specialization '{data.Name}' successfully.",
                        CreatedAt = DateTime.Now
                    });
                    db.SaveChanges();


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
                return InternalServerError(new Exception("Error while adding sub-specialization: " + ex.Message));
            }
        }




        // Get all SubSpecializations
        [HttpGet]
        [Route("api/admin/getAllSubSpecializations")]
        public IHttpActionResult GetAllSubSpecializations()
        {
            try
            {
                //Only select respective fields to send. No need to send specialization also
                var subs = db.SubSpecializations
                             .Select(s => new
                             {
                                 s.SubSpecializationId,
                                 s.Name,
                                 s.SpecializationId,
                                 SpecializationName = s.Specialization.Name
                             })
                             .ToList();

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
                return InternalServerError(new Exception("Error retrieving sub-specializations: " + ex.Message));
            }
        }




        // This will take specialization id and give all the associated subspecialist
        [HttpGet]
        [Route("api/admin/getSubSpecializationsBySpecialization/{id}")]
        public IHttpActionResult GetSubSpecializationsBySpecialization(int id)
        {
            try
            {
                // get all subspecialization where it matches given specialization id
                var subSpecializations = db.SubSpecializations
                                           .Where(ss => ss.SpecializationId == id)
                                           .Select(s => new   // only select necessary data
                                           {
                                               s.SubSpecializationId,
                                               s.Name,
                                               s.SpecializationId,
                                               SpecializationName = s.Specialization.Name
                                           })
                                          .ToList();

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
                return InternalServerError(new Exception("An error occurred while retrieving data: " + ex.Message));
            }
        }



        // Update an existing SubSpecialization
        [HttpPost]
        [Route("api/admin/updateSubSpecialization")]
        public IHttpActionResult UpdateSubSpecialization(SubSpecialization updatedData)
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
                var existing = db.SubSpecializations.Find(updatedData.SubSpecializationId);
                if (existing == null)
                    return NotFound();  //404 status code


                //checking if duplicate subspecialization given or not
                // condition: ignoring same subspecialization, and searching in same specialization
                bool isDuplicate = db.SubSpecializations
                                     .Any(s => s.SubSpecializationId != updatedData.SubSpecializationId &&
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
                db.SaveChanges();


                // Log: Successful update
                db.SystemLogs.Add(new SystemLog
                {
                    ActorType = "Admin",
                    ActorId = CustomFunctions.GetAdminUserIdFromToken(User),
                    Action = "Update Sub Specialization",
                    Details = $"Updated sub-specialization '{updatedData.Name}' successfully.",
                    CreatedAt = DateTime.Now
                });
                db.SaveChanges();

                return Ok(new
                {
                    success = true,
                    message = "Sub-specialization updated successfully.",
                    data = existing
                });
            }
            catch (Exception ex)
            {
                return InternalServerError(new Exception("Error updating sub-specialization: " + ex.Message));
            }
        }




        // Delete SubSpecialization by ID
        [HttpPost]
        [Route("api/admin/deleteSubSpecialization/{id}")]
        public IHttpActionResult DeleteSubSpecialization(int id)
        {
            try
            {
                // finding if the subspecialization exists or not
                var sub = db.SubSpecializations.FirstOrDefault(s => s.SubSpecializationId == id);

                if (sub == null)
                    return NotFound();   // if not found then send 404 status code

                db.SubSpecializations.Remove(sub);  // if found then delete that
                db.SaveChanges();


                // Log: Successful deletion
                db.SystemLogs.Add(new SystemLog
                {
                    ActorType = "Admin",
                    ActorId = CustomFunctions.GetAdminUserIdFromToken(User),
                    Action = "Delete Sub Specialization",
                    Details = $"Deleted sub-specialization '{sub.Name}' successfully.",
                    CreatedAt = DateTime.Now
                });
                db.SaveChanges();

                return Ok(new
                {
                    success = true,
                    message = "Sub-specialization deleted successfully.",
                    data = sub
                });
            }
            catch (Exception ex)
            {
                return InternalServerError(new Exception("Error deleting sub-specialization: " + ex.Message));
            }
        }


    }
}
