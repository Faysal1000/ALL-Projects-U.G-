using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Threading.Tasks;
using System.Web.Http;
using LifeinnovirorMentalHealthConsultency.Context.Tables;
using LifeinnovirorMentalHealthConsultency.Context;
using System.Data.Entity;

namespace LifeinnovirorMentalHealthConsultency.Controllers
{
    public class NotificationManagementController : ApiController
    {
        private readonly LifeinnovirorContext db;
        public NotificationManagementController()
        {
            db = new LifeinnovirorContext(); // Initializing the database
        }



        // this will get unread notification count for the logged-in admin
        [Authorize(Roles = "Admin")]
        [HttpGet]
        [Route("api/admin/getNotificationCount")]
        public async Task<IHttpActionResult> GetNotificationCount()
        {
            try
            {
                // get count of unread notifications for this admin
                int unreadCount = await db.Notifications
                    .Where(n => n.RecipientType == "Admin" && !n.Read)
                    .CountAsync();

                return Ok(new
                {
                    success = true,
                    message = "Unread notification count fetched successfully.",
                    data = new
                    {
                        unreadNotificationCount = unreadCount
                    }
                });
            }
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "Unexpected error while fetching Admin notification count.",
                    error = ex.Message
                });
            }
        }



        [Authorize(Roles = "Doctor")]
        [HttpGet]
        [Route("api/doctor/getNotificationCount")]
        public async Task<IHttpActionResult> GetDoctorNotificationCount()
        {
            try
            {
                int doctorId = CustomFunctions.GetDoctorUserIdFromToken(User);
                if (doctorId <= 0)
                {
                    return Content(HttpStatusCode.Unauthorized, new
                    {
                        success = false,
                        message = "Invalid doctor token."
                    });
                }

                var count = await db.Notifications
                                   .CountAsync(n => n.RecipientType == "Doctor" &&
                                                    n.RecipientId == doctorId &&
                                                    !n.Read);

                return Ok(new
                {
                    success = true,
                    message = "Doctor notification count retrieved.",
                    data = count
                });
            }
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "Unexpected error occurred while fetching doctor notification count.",
                    error = ex.Message
                });
            }
        }




        [Authorize(Roles = "Patient")]
        [HttpGet]
        [Route("api/patient/getNotificationCount")]
        public async Task<IHttpActionResult> GetPatientNotificationCount()
        {
            try
            {
                int patientId = CustomFunctions.GetPatientUserIdFromToken(User);
                if (patientId <= 0)
                {
                    return Content(HttpStatusCode.Unauthorized, new
                    {
                        success = false,
                        message = "Invalid patient token."
                    });
                }

                var count = await db.Notifications
                                   .CountAsync(n => n.RecipientType == "Patient" &&
                                                    n.RecipientId == patientId &&
                                                    !n.Read);

                return Ok(new
                {
                    success = true,
                    message = "Patient notification count retrieved.",
                    data = count
                });
            }
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "Unexpected error occurred while fetching patient notification count.",
                    error = ex.Message
                });
            }
        }




        [Authorize(Roles = "Admin")]
        [HttpGet]
        [Route("api/admin/getNotifications")]
        public async Task<IHttpActionResult> GetAdminNotifications()
        {
            try
            {
                // Get all notifications sorted by SentAt descending
                var allNotifications = await db.Notifications
                    .Where(n => n.RecipientType == "Admin")
                    .OrderByDescending(n => n.SentAt)
                    .ToListAsync();


                // Clone data to avoid showing updated status
                var responseData = allNotifications
                    .Select(n => new
                    {
                        n.NotificationId,
                        n.Message,
                        n.SentAt,
                        n.Read,
                        n.RecipientId
                    }).ToList();

                // Mark all unread notifications as read
                var unreadNotifications = allNotifications.Where(n => !n.Read).ToList();
                foreach (var notification in unreadNotifications)
                {
                    notification.Read = true;
                }
                await db.SaveChangesAsync();

                return Ok(new
                {
                    success = true,
                    message = "Admin notifications retrieved successfully.",
                    data = responseData
                });
            }
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "Unexpected error while retrieving admins notifications.",
                    error = ex.Message
                });
            }
        }




        [Authorize(Roles = "Doctor")]
        [HttpGet]
        [Route("api/doctor/getNotifications")]
        public async Task<IHttpActionResult> GetDoctorNotifications()
        {
            try
            {
                int doctorId = CustomFunctions.GetDoctorUserIdFromToken(User);
                if (doctorId <= 0)
                {
                    return Content(HttpStatusCode.Unauthorized, new
                    {
                        success = false,
                        message = "Invalid doctor token."
                    });
                }

                // Get all notifications sorted by SentAt descending
                var allNotifications = await db.Notifications
                    .Where(n => n.RecipientType == "Doctor" && n.RecipientId == doctorId)
                    .OrderByDescending(n => n.SentAt)
                    .ToListAsync();

                // Clone data to avoid showing updated status
                var responseData = allNotifications
                    .Select(n => new
                    {
                        n.NotificationId,
                        n.Message,
                        n.SentAt,
                        n.Read,
                        n.RecipientId
                    }).ToList();

                // Now update unread notifications to read
                var unreadNotifications = allNotifications.Where(n => !n.Read).ToList();
                foreach (var notification in unreadNotifications)
                {
                    notification.Read = true;
                }
                await db.SaveChangesAsync();

                return Ok(new
                {
                    success = true,
                    message = "Doctor notifications retrieved successfully.",
                    data = responseData
                });
            }
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "Unexpected error while retrieving doctor notifications.",
                    error = ex.Message
                });
            }
        }





        [Authorize(Roles = "Patient")]
        [HttpGet]
        [Route("api/patient/getNotifications")]
        public async Task<IHttpActionResult> GetPatientNotifications()
        {
            try
            {
                int patientId = CustomFunctions.GetPatientUserIdFromToken(User);
                if (patientId <= 0)
                {
                    return Content(HttpStatusCode.Unauthorized, new
                    {
                        success = false,
                        message = "Invalid patient token."
                    });
                }

                // Get all notifications sorted by SentAt descending
                var allNotifications = await db.Notifications
                    .Where(n => n.RecipientType == "Patient" && n.RecipientId == patientId)
                    .OrderByDescending(n => n.SentAt)
                    .ToListAsync();

                // Clone data to avoid showing updated status
                var responseData = allNotifications
                    .Select(n => new
                    {
                        n.NotificationId,
                        n.Message,
                        n.SentAt,
                        n.Read,
                        n.RecipientId
                    }).ToList();

                // Mark all unread notifications as read
                var unreadNotifications = allNotifications.Where(n => !n.Read).ToList();
                foreach (var notification in unreadNotifications)
                {
                    notification.Read = true;
                }
                await db.SaveChangesAsync();

                return Ok(new
                {
                    success = true,
                    message = "Patient notifications retrieved successfully.",
                    data = responseData
                });
            }
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "Unexpected error while retrieving patients notifications.",
                    error = ex.Message
                });
            }
        }

    }

}
