using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Security.Claims;
using System.Web.Http;
using LifeinnovirorMentalHealthConsultency.Context;
using LifeinnovirorMentalHealthConsultency.Context.Tables;

namespace LifeinnovirorMentalHealthConsultency.Controllers
{
    public class AdminController : ApiController
    {
        private LifeinnovirorContext db;    // Creating private db object to manupulate data
        public AdminController()
        {
            db = new LifeinnovirorContext(); // Initializing the database in constructor 
        }

        [Authorize(Roles = "Admin")]
        [HttpPost]
        [Route("api/admin/addSpecialization")]
        public IHttpActionResult AddSpecialization(Specialization data)
        {
            //Try-catch block to handle unintended errors
            try
            {
                //
                if (ModelState.IsValid)
                {
                    db.Specializations.Add(data);
                    db.SaveChanges();
                    return Ok(new
                    {
                        success = true,
                        message = "Specialization added successfully.",
                        data = data
                    });
                }

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
                return InternalServerError(new Exception("Error while adding specialization: " + ex.Message));
            }
        }



        [Authorize(Roles = "Admin")]
        [HttpGet]
        [Route("api/admin/getAllSpecializations")]
        public IHttpActionResult GetAllSpecializations()
        {
            try
            {
                var specializations = db.Specializations.ToList();

                if (specializations == null || !specializations.Any())
                {
                    return Ok(new
                    {
                        success = true,
                        message = "No specializations found.",
                        data = new List<object>()
                    });
                }

                return Ok(new
                {
                    success = true,
                    message = "Specializations retrieved successfully.",
                    data = specializations
                });
            }
            catch (Exception ex)
            {
                return InternalServerError(new Exception("Error retrieving specializations: " + ex.Message));
            }
        }



    }
}
