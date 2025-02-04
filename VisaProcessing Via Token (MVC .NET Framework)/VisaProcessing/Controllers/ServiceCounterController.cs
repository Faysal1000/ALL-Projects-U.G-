using Antlr.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Mvc;
using VisaProcessing.Entity_Framework;

namespace VisaProcessing.Controllers
{
    public class ServiceCounterController : Controller
    {
        VisaProcessingContext db = new VisaProcessingContext();
        [HttpGet]
        public ActionResult CallCustomer()
        {
            return View();
        }
        [HttpPost]
        public ActionResult CallCustomer(int counter)
        {
            var todayDate = DateTime.Now.ToString("dd-MMM-yy");
            var log = (from data in db.Tokens
                       where data.TokenNo.Contains(todayDate) &
                             data.Service==false
                       select data).ToList();

            if (log.Count == 0)
            {
                TempData["msg"] = "No Token Remains to Process";
                return View();
            }

            var medlog = (from data in log
                          where data.TokenNo.StartsWith("Med-")
                          orderby data.TokenNo
                          select data).ToList();
            var TRlog = (from data in log
                          where data.TokenNo.StartsWith("TR-")
                         orderby data.TokenNo
                         select data).ToList();
            var BSlog = (from data in log
                          where data.TokenNo.StartsWith("BS-")
                         orderby data.TokenNo
                         select data).ToList();
            var GOlog = (from data in log
                          where data.TokenNo.StartsWith("GO-")
                         orderby data.TokenNo
                         select data).ToList();

            var largestList = medlog;
            if (TRlog.Count > largestList.Count)
            {
                largestList = TRlog;
            }
            if (BSlog.Count > largestList.Count)
            {
                largestList = BSlog;
            }
            if (GOlog.Count > largestList.Count)
            {
                largestList = GOlog;
            }

            switch (counter)
            {
                case 0:
                    if (medlog.Count > 0) {
                        medlog[0].Service = true;
                        TempData["token"] = medlog[0].TokenNo;
                    }
                    else
                    {
                        TempData["token"] = largestList[0].TokenNo;
                        largestList[0].Service = true;
                    }
                    break;
                case 1:
                    if (TRlog.Count > 0)
                    {
                        TRlog[0].Service = true;
                        TempData["token"] = TRlog[0].TokenNo;
                    }
                    else
                    {
                        TempData["token"] = largestList[0].TokenNo;
                        largestList[0].Service = true;
                    }
                    break;
                case 2:
                    if (BSlog.Count > 0)
                    {
                        BSlog[0].Service = true;
                        TempData["token"] = BSlog[0].TokenNo;
                    }
                    else
                    {
                        TempData["token"] = largestList[0].TokenNo;
                        largestList[0].Service = true;
                    }
                    break;
                case 3:
                    if (GOlog.Count > 0)
                    {
                        GOlog[0].Service = true;
                        TempData["token"] = GOlog[0].TokenNo;
                    }
                    else
                    {
                        TempData["token"] = largestList[0].TokenNo;
                        largestList[0].Service = true;
                    }
                    break;
            }
            db.SaveChanges();
            return RedirectToAction("Servicing");
        }
        public ActionResult Servicing()
        {
            ViewBag.token=TempData["token"];
            return View();
        }
    }
}