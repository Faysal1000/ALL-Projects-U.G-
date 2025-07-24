using System;
using System.Collections.Generic;
using System.Data.Entity;
using System.Linq;
using System.Net;
using System.Threading.Tasks;
using System.Web.Http;
using LifeinnovirorMentalHealthConsultency.Context;
using LifeinnovirorMentalHealthConsultency.Context.Tables;
using LifeinnovirorMentalHealthConsultency.Functional_Class;
using LifeinnovirorMentalHealthConsultency.Models;

namespace LifeinnovirorMentalHealthConsultency.Controllers.AdminControllers
{
    [Authorize(Roles = "Admin")]
    public class AdminDoctorRegistrationController : ApiController
    {
        private readonly LifeinnovirorContext db;
        public AdminDoctorRegistrationController()
        {
            db = new LifeinnovirorContext(); // Initializing the database 
        }


        [HttpGet]
        [Route("api/Admin/getRequestedDoctorRegistration")]
        public async Task<IHttpActionResult> GetRequestedDoctorRegistration()
        {
            try
            {
                var pendingDoctors = await db.Doctors
                    .Where(d => d.Status == "Pending" || d.Status=="Interview")
                    .Select(d => new
                    {
                        d.DoctorId,
                        d.FullName,
                        d.Email,
                        d.PhoneNumber,
                        d.Qualifications,
                        d.ExperienceSummary,
                        d.YearsOfExperience,
                        d.Status,
                        d.CreatedAt,
                    })
                    .OrderByDescending(d => d.Status) // interview first
                    .ToListAsync();

                return Ok(new
                {
                    success = true,
                    message = "Pending doctor registration requests retrieved successfully.",
                    data = pendingDoctors
                });
            }
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "An error occurred while retrieving doctor registration requests.",
                    error = ex.Message
                });
            }
        }


        [HttpGet]
        [Route("api/Admin/getRequestedDoctorRegistrationCount")]
        public async Task<IHttpActionResult> GetRequestedDoctorRegistrationCount()
        {
            try
            {
                var count = await db.Doctors.CountAsync(d => d.Status == "Pending" || d.Status=="Interview");

                return Ok(new
                {
                    success = true,
                    message = "Successfully retrieved pending doctor registration count.",
                    count = count
                });
            }
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "An error occurred while retrieving count.",
                    error = ex.Message
                });
            }
        }




        [HttpPost]
        [Route("api/Admin/updateDoctorStatus")]
        public async Task<IHttpActionResult> UpdateDoctorStatus(DoctorStatusUpdateRequestModel model)
        {
            try
            {
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


                // Find doctor
                var doctor = await db.Doctors.FirstOrDefaultAsync(d => d.DoctorId == model.DoctorId);
                if (doctor == null)
                {
                    return Content(HttpStatusCode.NotFound, new
                    {
                        success = false,
                        message = "Doctor not found."
                    });
                }

                // Update status
                doctor.Status = model.NewStatus;
                doctor.UpdatedAt = DateTime.Now;

                // Create notification
                string notificationMessage = "";
                string mailStatusReport="";

                if (model.NewStatus == "Interview")
                {
                    notificationMessage = CustomVariables.doctorInterviewNotificationMessage;
                    mailStatusReport = EmailManagement.DoctorInterviewEmail(doctor.FullName, doctor.Email);

                }
                else if (model.NewStatus == "Rejected")
                {
                    notificationMessage = CustomVariables.doctorRejectNotificationMessage;
                    mailStatusReport = EmailManagement.DoctorRejectedEmail(doctor.FullName, doctor.Email);

                }
                else if (model.NewStatus == "Approved")
                {
                    notificationMessage = CustomVariables.doctorApprovedNotificationMessage;
                    mailStatusReport = EmailManagement.DoctorApprovedEmail(doctor.FullName, doctor.Email);

                }
                if (!string.IsNullOrEmpty(notificationMessage))
                {
                    var notification = new Notification
                    {
                        RecipientType = "Doctor",
                        RecipientId = model.DoctorId,
                        Message = notificationMessage,
                        SentAt = DateTime.Now,
                        Read = false
                    };
                    db.Notifications.Add(notification);
                }


                // Log
                db.SystemLogs.Add(new SystemLog
                {
                    ActorType = "Admin",
                    ActorId = CustomFunctions.GetAdminUserIdFromToken(User),
                    Action = "Update Doctor Registration Status",
                    Details = $"Doctor, Id = {model.DoctorId}, Registration Status set to {model.NewStatus}",
                    CreatedAt = DateTime.Now
                });
                await db.SaveChangesAsync();

                // Return response
                return Ok(new
                {
                    success = true,
                    message = $"Doctor status updated to '{model.NewStatus}' successfully.",
                    mailStatusReport = mailStatusReport
                });
            }
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "An error occurred while updating doctor status.",
                    error = ex.Message
                });
            }
        }
    }
}
