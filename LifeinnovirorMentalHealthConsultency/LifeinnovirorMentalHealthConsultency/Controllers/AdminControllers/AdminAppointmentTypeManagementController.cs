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
    public class AdminAppointmentTypeManagementController : ApiController
    {
        private readonly LifeinnovirorContext db;   
        public AdminAppointmentTypeManagementController()
        {
            db = new LifeinnovirorContext(); // Initializing the database
        }



        [Authorize(Roles = "Admin")]
        [HttpPost]
        [Route("api/admin/addAppointmentType")]
        public async Task<IHttpActionResult> AddAppointmentType(AppointmentType model)
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
                var existing = await db.AppointmentTypes
                                     .FirstOrDefaultAsync(f => f.Name.Trim().ToLower() == incomingAppointmentName);

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

                //add successfull addition logs
                db.SystemLogs.Add(new SystemLog
                {
                    ActorType = "Admin",
                    ActorId = CustomFunctions.GetAdminUserIdFromToken(User),
                    Action = "Add Appointment Type",
                    Details = $"Added appointment type '{model.Name}' with cost {model.Cost}.",
                    CreatedAt = DateTime.Now
                });
                await db.SaveChangesAsync();

                return Ok(new
                {
                    success = true,
                    message = "Appointment type added successfully.",
                    data = model
                });
            }
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "Unexpected error while adding appointment type.",
                    error = ex.Message
                });
            }
        }


        [Authorize(Roles = "Admin")]
        [HttpPost]
        [Route("api/admin/updateAppointmentType")]
        public async Task<IHttpActionResult> UpdateAppointmentType(AppointmentType updatedType)
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
                var existing = await db.AppointmentTypes.FindAsync(updatedType.AppointmentTypeId);
                if (existing == null)
                {
                    return NotFound();
                }

                //update data
                existing.Name = updatedType.Name;
                existing.Cost = updatedType.Cost;

                // add update appointment type logs
                db.SystemLogs.Add(new SystemLog
                {
                    ActorType = "Admin",
                    ActorId = CustomFunctions.GetAdminUserIdFromToken(User),
                    Action = "Update Appointment Type",
                    Details = $"Updated appointment type ID {existing.AppointmentTypeId}.",
                    CreatedAt = DateTime.Now
                });
                await db.SaveChangesAsync();

                return Ok(new
                {
                    success = true,
                    message = "Appointment type updated successfully.",
                    data = existing
                });
            }
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "Unexpected error while updating appointment type.",
                    error = ex.Message
                });
            }
        }


        [Authorize(Roles = "Admin")]
        [HttpPost]
        [Route("api/admin/deleteAppointmentType")]
        public async Task<IHttpActionResult> DeleteAppointmentType(int id)
        {
            try
            {
                //check if appointment type exists or not
                var type = await db.AppointmentTypes.FindAsync(id);
                if (type == null)
                {
                    return NotFound();
                }

                db.AppointmentTypes.Remove(type);

                // add deletion system log
                db.SystemLogs.Add(new SystemLog
                {
                    ActorType = "Admin",
                    ActorId = CustomFunctions.GetAdminUserIdFromToken(User),
                    Action = "Delete Appointment Type",
                    Details = $"Deleted appointment type '{type.Name}' with ID {id}.",
                    CreatedAt = DateTime.Now
                });
                await db.SaveChangesAsync();

                return Ok(new
                {
                    success = true,
                    message = "Appointment type deleted successfully.",
                    data = type
                });
            }
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "Unexpected error while deleting appointment type.",
                    error = ex.Message
                });
            }
        }




        // no authorization cause it need to be accessed by home page
        [HttpGet]
        [Route("api/getAppointmentTypes")]
        public async Task<IHttpActionResult> GetAppointmentTypes()
        {
            try
            {
                var types = await db.AppointmentTypes.ToListAsync();

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
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "Unexpected error while retriving appointment type.",
                    error = ex.Message
                });
            }
        }

    }
}
