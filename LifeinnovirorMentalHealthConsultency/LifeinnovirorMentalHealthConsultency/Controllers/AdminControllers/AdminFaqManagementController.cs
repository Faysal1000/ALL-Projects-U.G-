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
    public class AdminFaqManagementController : ApiController
    {
        private readonly LifeinnovirorContext db;    // Creating private db object to manupulate data
        public AdminFaqManagementController()
        {
            db = new LifeinnovirorContext(); // Initializing the database in constructor 
        }



        [Authorize(Roles = "Admin")]
        [HttpPost]
        [Route("api/admin/createFaq")]
        public IHttpActionResult CreateFaq(FAQ model)
        {
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
                        data = model
                    });
                }

                // Normalize input question: lowercase and trimmed
                string incomingQuestion = model.Question.Trim().ToLower();

                // Check if a FAQ with same normalized question already exists
                var existing = db.FAQs
                    .FirstOrDefault(f => f.Question.Trim().ToLower() == incomingQuestion);

                if (existing != null)
                {
                    return Content(HttpStatusCode.Conflict, new
                    {
                        success = false,
                        message = "This FAQ question already exists.",
                        data = model
                    });
                }


                model.CreatedAt = DateTime.Now;
                model.UpdatedAt = DateTime.Now;

                db.FAQs.Add(model);
                db.SaveChanges();

                // Logs: successfull addition
                db.SystemLogs.Add(new SystemLog
                {
                    ActorType = "Admin",
                    ActorId = CustomFunctions.GetAdminUserIdFromToken(User),
                    Action = "Create FAQ",
                    Details = $"Created FAQ with Question: '{model.Question}'",
                    CreatedAt = DateTime.Now
                });
                db.SaveChanges();

                return Ok(new
                {
                    success = true,
                    message = "FAQ created successfully.",
                    data = model
                });
            }
            catch (Exception ex)
            {
                return InternalServerError(new Exception("An error occurred while creating FAQs: " + ex.Message));
            }
        }


        //this can be access by anyone cause it will be in home page. 
        [HttpGet]
        [Route("api/getFaqs")]
        public IHttpActionResult GetAllFaqs()
        {
            try
            {
                var faqs = db.FAQs.OrderByDescending(f => f.UpdatedAt).ToList();

                // If no faqs found then it will send success message with the message
                if (faqs == null || !faqs.Any())
                {
                    return Ok(new
                    {
                        success = true,
                        message = "No FAQs found.",
                        data = new List<object>()
                    });
                }

                // If faqs found then it will send all faqs objects as list
                return Ok(new
                {
                    success = true,
                    message = "FAQs retrieved successfully.",
                    data = faqs
                });
            }
            catch (Exception ex)
            {
                return InternalServerError(new Exception("An error occurred while retrieving FAQs: " + ex.Message));
            }
        }



        // This will take updated FAQ data and update the existing FAQ
        [Authorize(Roles = "Admin")]
        [HttpPost]
        [Route("api/admin/updateFAQ")]
        public IHttpActionResult UpdateFAQ(FAQ updatedFaq)
        {
            try
            {
                // if the data is invalid then send error message
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
                        data = updatedFaq
                    });
                }

                // check if the FAQ exists in database
                var existingFaq = db.FAQs.Find(updatedFaq.FaqId);
                if (existingFaq == null)
                {
                    return NotFound(); // if not found then send 404
                }

                // update fields
                existingFaq.Question = updatedFaq.Question;
                existingFaq.Answer = updatedFaq.Answer;
                existingFaq.UpdatedAt = DateTime.Now;
                db.SaveChanges();

                // Log: Successful update
                db.SystemLogs.Add(new SystemLog
                {
                    ActorType = "Admin",
                    ActorId = CustomFunctions.GetAdminUserIdFromToken(User),
                    Action = "Update FAQ",
                    Details = $"Updated FAQ (ID: {updatedFaq.FaqId}) successfully.",
                    CreatedAt = DateTime.Now
                });
                db.SaveChanges();

                return Ok(new
                {
                    success = true,
                    message = "FAQ updated successfully.",
                    data = existingFaq
                });
            }
            catch (Exception ex)
            {
                return InternalServerError(new Exception("An error occurred while updating FAQ: " + ex.Message));
            }
        }


        // This will take faq id and delete the existing FAQ
        [Authorize(Roles = "Admin")]
        [HttpPost]
        [Route("api/admin/deleteFaq/{id}")]
        public IHttpActionResult DeleteFaq(int id)
        {
            try
            {
                // check if faq exists or not
                var faq = db.FAQs.FirstOrDefault(f => f.FaqId == id);
                if (faq == null)
                    return NotFound(); //404 status code

                db.FAQs.Remove(faq);
                db.SaveChanges();

                //add deletion logs
                db.SystemLogs.Add(new SystemLog
                {
                    ActorType = "Admin",
                    ActorId = CustomFunctions.GetAdminUserIdFromToken(User),
                    Action = "Delete FAQ",
                    Details = $"Deleted FAQ with Question: '{faq.Question}'",
                    CreatedAt = DateTime.Now
                });
                db.SaveChanges();

                return Ok(new
                {
                    success = true,
                    message = "FAQ deleted successfully.",
                    data = faq
                });
            }
            catch (Exception ex)
            {
                return InternalServerError(new Exception("An error occurred while deleting FAQ: " + ex.Message));
            }
        }

    }
}
