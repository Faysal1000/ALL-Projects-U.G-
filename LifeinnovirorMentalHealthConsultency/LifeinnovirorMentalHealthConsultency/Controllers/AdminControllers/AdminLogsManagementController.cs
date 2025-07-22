using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Web.Http;
using LifeinnovirorMentalHealthConsultency.Context;

namespace LifeinnovirorMentalHealthConsultency.Controllers.AdminControllers
{
    [Authorize(Roles = "Admin")]
    public class AdminLogsManagementController : ApiController
    {
        private readonly LifeinnovirorContext db;    // Creating private db object to manupulate data
        public AdminLogsManagementController()
        {
            db = new LifeinnovirorContext(); // Initializing the database in constructor 
        }



        [HttpGet]
        [Route("api/Admin/getAllLogs")]
        public IHttpActionResult GetAllLogs()
        {
            try
            {
                //latest first
                var logs = db.SystemLogs.OrderByDescending(log => log.CreatedAt).ToList();

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
                return InternalServerError(new Exception("An error occurred while retrieving all logs: " + ex.Message));
            }
        }



        [HttpGet]
        [Route("api/Admin/getAdminLogs")]
        public IHttpActionResult GetAdminLogs()
        {
            try
            {
                // only admin logs and latest first
                var logs = db.SystemLogs
                             .Where(log => log.ActorType == "Admin")
                             .OrderByDescending(log => log.CreatedAt)
                             .ToList();

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
                return InternalServerError(new Exception("An error occurred while retrieving admin logs: " + ex.Message));
            }
        }



        [HttpGet]
        [Route("api/Admin/getDoctorLogs")]
        public IHttpActionResult GetDcotorLogs()
        {
            try
            {
                // only doctor logs and latest first
                var logs = db.SystemLogs
                             .Where(log => log.ActorType == "Doctor")
                             .OrderByDescending(log => log.CreatedAt)
                             .ToList();

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
                return InternalServerError(new Exception("An error occurred while retrieving doctor logs: " + ex.Message));
            }
        }



        [HttpGet]
        [Route("api/Admin/getPatientLogs")]
        public IHttpActionResult GetPatientLogs()
        {
            try
            {
                // only patient logs and latest first
                var logs = db.SystemLogs
                             .Where(log => log.ActorType == "Patient")
                             .OrderByDescending(log => log.CreatedAt)
                             .ToList();

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
                return InternalServerError(new Exception("An error occurred while retrieving patient logs: " + ex.Message));
            }
        }

    }
}
