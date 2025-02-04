using System;
using System.Collections.Generic;
using System.Data.Entity;
using System.Linq;
using System.Web;
using System.Web.Mvc;
using VisaProcessing.Entity_Framework;
using VisaProcessing.Entity_Framework.Table;

namespace VisaProcessing.Controllers
{
    public class TokenController : Controller
    {
        VisaProcessingContext db = new VisaProcessingContext();
        [HttpGet]
        public ActionResult TokenGenerate()
        { 
            return View();
        }
        [HttpPost]
        public ActionResult TokenGenerate(string visaPurpose)
        {
            string token = "";
            var log = (from data in db.TokenCounters
                       where DbFunctions.TruncateTime(data.date) == DbFunctions.TruncateTime(DateTime.Now)
                       select data).SingleOrDefault();

            if (log == null)
            {
                log = new TokenCounter
                {
                    MedCounter = 0,
                    TrCounter = 0,
                    BsCounter = 0,
                    GoCounter = 0,
                    TotalCounter = 0,
                    date = DateTime.Now
                };
                db.TokenCounters.Add(log);
                db.SaveChanges();  
            }

            switch (visaPurpose)
            {
                case "Medical":
                    if (log.MedCounter < 25)
                    {
                        log.MedCounter++;
                        log.TotalCounter++;
                        token = "Med-" + log.TotalCounter+ " date: "+ log.date;
                        db.Tokens.Add(new Tokens{TokenNo = token, Service = false});

                    }
                    break;
                case "Tourist":
                    if (log.TrCounter < 25)
                    {
                        log.TrCounter++;
                        log.TotalCounter++;
                        token = "TR-" + log.TotalCounter + " date: " + log.date;
                        db.Tokens.Add(new Tokens { TokenNo = token, Service = false });
                    }
                    break;
                case "Business":
                    if (log.BsCounter < 25)
                    {
                        log.BsCounter++;
                        log.TotalCounter++;
                        token = "BS-" + log.TotalCounter + " date: " + log.date;
                        db.Tokens.Add(new Tokens { TokenNo = token, Service = false });
                    }
                    break;
                case "Govt":
                    if (log.GoCounter < 25)
                    {
                        log.GoCounter++;
                        log.TotalCounter++;
                        token = "GO-" + log.TotalCounter + " date: " + log.date;
                        db.Tokens.Add(new Tokens { TokenNo = token, Service = false });
                    }
                    break;
            }

            if (!string.IsNullOrEmpty(token))
            {
                db.SaveChanges();
                TempData["token"] = token; 
                return RedirectToAction("ViewToken");
            }
            TempData["msg"] = "No available token";  
            return View();
        }
        public ActionResult ViewToken()
        {
            ViewBag.token = TempData["token"];
            return View();
        }

    }
}