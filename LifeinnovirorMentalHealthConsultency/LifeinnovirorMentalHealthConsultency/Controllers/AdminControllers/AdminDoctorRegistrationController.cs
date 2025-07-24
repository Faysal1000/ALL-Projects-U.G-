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
        public async Task<IHttpActionResult> UpdateDoctorStatus(int doctorId, string newStatus)
        {
            try
            {
                // Validate status (Pending|Approved|Interview|Rejected)
                var validStatuses = new List<string> { "Approved", "Interview", "Rejected" };
                if (!validStatuses.Contains(newStatus))
                {
                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = "Invalid status value. Allowed values: Approved, Interview, Rejected."
                    });
                }

                // Find doctor
                var doctor = await db.Doctors.FirstOrDefaultAsync(d => d.DoctorId == doctorId);
                if (doctor == null)
                {
                    return Content(HttpStatusCode.NotFound, new
                    {
                        success = false,
                        message = "Doctor not found."
                    });
                }

                // Update status
                doctor.Status = newStatus;
                doctor.UpdatedAt = DateTime.Now;

                // Create notification
                string notificationMessage = "";
                string mailStatusReport="";

                if (newStatus == "Interview")
                {
                    notificationMessage = CustomVariables.doctorInterviewNotificationMessage;
                    mailStatusReport = EmailManagement.DoctorInterviewEmail(doctor.FullName, doctor.Email);

                }
                else if (newStatus == "Rejected")
                {
                    notificationMessage = CustomVariables.doctorRejectNotificationMessage;
                    mailStatusReport = EmailManagement.DoctorRejectedEmail(doctor.FullName, doctor.Email);

                }
                else if (newStatus == "Approved")
                {
                    notificationMessage = CustomVariables.doctorApprovedNotificationMessage;
                    mailStatusReport = EmailManagement.DoctorApprovedEmail(doctor.FullName, doctor.Email);

                }
                if (!string.IsNullOrEmpty(notificationMessage))
                {
                    var notification = new Notification
                    {
                        RecipientType = "Doctor",
                        RecipientId = doctorId,
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
                    Details = $"Doctor, Id = {doctorId}, Registration Status set to {newStatus}",
                    CreatedAt = DateTime.Now
                });
                await db.SaveChangesAsync();

                // Return response
                return Ok(new
                {
                    success = true,
                    message = $"Doctor status updated to '{newStatus}' successfully.",
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
