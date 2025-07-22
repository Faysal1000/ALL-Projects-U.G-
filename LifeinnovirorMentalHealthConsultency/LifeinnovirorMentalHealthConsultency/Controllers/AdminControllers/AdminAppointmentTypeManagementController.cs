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
    public class AdminAppointmentTypeManagementController : ApiController
    {
        private readonly LifeinnovirorContext db;    // Creating private db object to manupulate data
        public AdminAppointmentTypeManagementController()
        {
            db = new LifeinnovirorContext(); // Initializing the database in constructor 
        }



        [Authorize(Roles = "Admin")]
        [HttpPost]
        [Route("api/admin/addAppointmentType")]
        public IHttpActionResult AddAppointmentType(AppointmentType model)
        {
            try
            {
                //if invalid then send model error
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
                        data = model
                    });
                }

                // Normalize input name: lowercase and trimmed
                string incomingAppointmentName = model.Name.Trim().ToLower();

                // Check if a Appointment with same name already exists
                var existing = db.AppointmentTypes
                    .FirstOrDefault(f => f.Name.Trim().ToLower() == incomingAppointmentName);

                if (existing != null)
                {
                    return Content(HttpStatusCode.Conflict, new
                    {
                        success = false,
                        message = "This Appointment Type already exists.",
                        data = model
                    });
                }

                db.AppointmentTypes.Add(model);
                db.SaveChanges();

                //add successfull addition logs
                db.SystemLogs.Add(new SystemLog
                {
                    ActorType = "Admin",
                    ActorId = CustomFunctions.GetAdminUserIdFromToken(User),
                    Action = "Add Appointment Type",
                    Details = $"Added appointment type '{model.Name}' with cost {model.Cost}.",
                    CreatedAt = DateTime.Now
                });
                db.SaveChanges();

                return Ok(new
                {
                    success = true,
                    message = "Appointment type added successfully.",
                    data = model
                });
            }
            catch (Exception ex)
            {
                return InternalServerError(new Exception("An error occurred while adding appointment type: " + ex.Message));
            }
        }


        [Authorize(Roles = "Admin")]
        [HttpPost]
        [Route("api/admin/updateAppointmentType")]
        public IHttpActionResult UpdateAppointmentType(AppointmentType updatedType)
        {
            try
            {
                // give error if invalid data
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
                        data = updatedType
                    });
                }

                // check if the appointmenttype exists or not
                var existing = db.AppointmentTypes.Find(updatedType.AppointmentTypeId);
                if (existing == null)
                {
                    return NotFound();
                }

                existing.Name = updatedType.Name;
                existing.Cost = updatedType.Cost;
                db.SaveChanges();

                // add update appointment type logs
                db.SystemLogs.Add(new SystemLog
                {
                    ActorType = "Admin",
                    ActorId = CustomFunctions.GetAdminUserIdFromToken(User),
                    Action = "Update Appointment Type",
                    Details = $"Updated appointment type ID {existing.AppointmentTypeId}.",
                    CreatedAt = DateTime.Now
                });
                db.SaveChanges();

                return Ok(new
                {
                    success = true,
                    message = "Appointment type updated successfully.",
                    data = existing
                });
            }
            catch (Exception ex)
            {
                return InternalServerError(new Exception("An error occurred while updating appointment type: " + ex.Message));
            }
        }


        [Authorize(Roles = "Admin")]
        [HttpPost]
        [Route("api/admin/deleteAppointmentType")]
        public IHttpActionResult DeleteAppointmentType(int id)
        {
            try
            {
                //check if appointment type exists or not
                var type = db.AppointmentTypes.Find(id);
                if (type == null)
                {
                    return NotFound();
                }

                db.AppointmentTypes.Remove(type);
                db.SaveChanges();

                // add deletion system log
                db.SystemLogs.Add(new SystemLog
                {
                    ActorType = "Admin",
                    ActorId = CustomFunctions.GetAdminUserIdFromToken(User),
                    Action = "Delete Appointment Type",
                    Details = $"Deleted appointment type '{type.Name}' with ID {id}.",
                    CreatedAt = DateTime.Now
                });
                db.SaveChanges();

                return Ok(new
                {
                    success = true,
                    message = "Appointment type deleted successfully.",
                    data = type
                });
            }
            catch (Exception ex)
            {
                return InternalServerError(new Exception("An error occurred while deleting appointment type: " + ex.Message));
            }
        }




        // no authorization cause it need to be accessed by home page
        [HttpGet]
        [Route("api/getAppointmentTypes")]
        public IHttpActionResult GetAppointmentTypes()
        {
            try
            {
                var types = db.AppointmentTypes.ToList();

                // If no type found then it will send success message with the message
                if (types == null || !types.Any())
                {
                    return Ok(new
                    {
                        success = true,
                        message = "No Appointment Types found.",
                        data = new List<object>()
                    });
                }
                return Ok(new
                {
                    success = true,
                    message = "Appointment types retrieved successfully.",
                    data = types
                });
            }
            catch (Exception ex)
            {
                return InternalServerError(new Exception("An error occurred while retrieving appointment types: " + ex.Message));
            }
        }

    }
}
