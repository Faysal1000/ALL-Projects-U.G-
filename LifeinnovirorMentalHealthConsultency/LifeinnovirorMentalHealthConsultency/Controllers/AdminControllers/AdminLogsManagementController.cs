using System;
using System.Collections.Generic;
using System.Data.Entity;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Threading.Tasks;
using System.Web.Http;
using LifeinnovirorMentalHealthConsultency.Context;

namespace LifeinnovirorMentalHealthConsultency.Controllers.AdminControllers
{
    [Authorize(Roles = "Admin")]
    public class AdminLogsManagementController : ApiController
    {
        private readonly LifeinnovirorContext db;  
        public AdminLogsManagementController()
        {
            db = new LifeinnovirorContext(); // Initializing the database
        }



        [HttpGet]
        [Route("api/Admin/getAllLogs")]
        public async Task<IHttpActionResult> GetAllLogs()
        {
            try
            {
                //latest first
                var logs = await db.SystemLogs.OrderByDescending(log => log.CreatedAt).ToListAsync();

                // If no logs found then it will send success message with the message
                if (logs == null || !logs.Any())
                {
                    return Ok(new
                    {
                        success = true,
                        message = "No logs found.",
                        data = new List<object>()
                    });
                }
                return Ok(new
                {
                    success = true,
                    message = "All logs retrieved successfully.",
                    data = logs
                });
            }
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "An error occurred while retriving logs",
                    error = ex.Message
                });
            }
        }



        [HttpGet]
        [Route("api/Admin/getAdminLogs")]
        public async Task<IHttpActionResult> GetAdminLogs()
        {
            try
            {
                // only admin logs and latest first
                var logs = await db.SystemLogs
                                 .Where(log => log.ActorType == "Admin")
                                 .OrderByDescending(log => log.CreatedAt)
                                 .ToListAsync();

                // If no logs found then it will send success message with the message
                if (logs == null || !logs.Any())
                {
                    return Ok(new
                    {
                        success = true,
                        message = "No Admin logs found.",
                        data = new List<object>()
                    });
                }
                return Ok(new
                {
                    success = true,
                    message = "Admin logs retrieved successfully.",
                    data = logs
                });
            }
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "An error occurred while retriving admin logs",
                    error = ex.Message
                });
            }
        }



        [HttpGet]
        [Route("api/Admin/getDoctorLogs")]
        public async Task<IHttpActionResult> GetDcotorLogs()
        {
            try
            {
                // only doctor logs and latest first
                var logs = await db.SystemLogs
                                 .Where(log => log.ActorType == "Doctor")
                                 .OrderByDescending(log => log.CreatedAt)
                                 .ToListAsync();

                // If no logs found then it will send success message with the message
                if (logs == null || !logs.Any())
                {
                    return Ok(new
                    {
                        success = true,
                        message = "No Doctor logs found.",
                        data = new List<object>()
                    });
                }
                return Ok(new
                {
                    success = true,
                    message = "Doctor logs retrieved successfully.",
                    data = logs
                });
            }
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "An error occurred while retriving doctor logs",
                    error = ex.Message
                });
            }
        }



        [HttpGet]
        [Route("api/Admin/getPatientLogs")]
        public async Task<IHttpActionResult> GetPatientLogs()
        {
            try
            {
                // only patient logs and latest first
                var logs = await db.SystemLogs
                                 .Where(log => log.ActorType == "Patient")
                                 .OrderByDescending(log => log.CreatedAt)
                                 .ToListAsync();

                // If no logs found then it will send success message with the message
                if (logs == null || !logs.Any())
                {
                    return Ok(new
                    {
                        success = true,
                        message = "No Patient logs found.",
                        data = new List<object>()
                    });
                }
                return Ok(new
                {
                    success = true,
                    message = "Patient logs retrieved successfully.",
                    data = logs
                });
            }
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "An error occurred while retriving patient logs",
                    error = ex.Message
                });
            }
        }

    }
}
