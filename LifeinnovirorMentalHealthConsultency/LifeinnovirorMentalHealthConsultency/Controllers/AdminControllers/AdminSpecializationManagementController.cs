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


        //This will take Specialization data inputed by admin and add it to the database
        [HttpPost]
        [Route("api/admin/addSpecialization")]
        public IHttpActionResult AddSpecialization(Specialization data)
        {
            //Try-catch block to handle unintended errors
            try
            {
                // If the Data is valid then it will add to database
                if (ModelState.IsValid)
                {
                    // Check if a specialization with the same name already exists (case-insensitive)
                    var exists = db.Specializations
                                   .Any(s => s.Name.ToLower().Trim() == data.Name.ToLower().Trim());

                    // If exists then send HTTPconflict code
                    if (exists)
                    {
                        return Content(HttpStatusCode.Conflict, new   // 409 if not conflict
                        {
                            success = false,
                            message = "This specialization already exists.",
                            data = data
                        });
                    }

                    db.Specializations.Add(data);
                    db.SaveChanges();

                    return Ok(new
                    {
                        success = true,
                        message = "Specialization added successfully.",
                        data = data
                    });
                }


                // If data is invalid then the error message and the from data will sent to frontend
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


        // This will take an id of the specialization object and delete it
        [HttpPost]
        [Route("api/admin/deleteSpecialization/{id}")]
        public IHttpActionResult DeleteSpecialization(int id)
        {
            //Try-catch block to handle unintended errors
            try
            {
                // Find if the existed id have data to delete 
                var specialization = db.Specializations.FirstOrDefault(s => s.SpecializationId == id);

                if (specialization == null)
                {
                    return NotFound(); // 404 if not found
                }

                // If found then delete
                db.Specializations.Remove(specialization);
                db.SaveChanges();

                return Ok(new
                {
                    success = true,
                    message = "Specialization deleted successfully.",
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
