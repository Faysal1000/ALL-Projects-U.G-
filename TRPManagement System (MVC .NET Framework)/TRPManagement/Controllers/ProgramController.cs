using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Mvc;
using TRPManagement.DTOs;
using TRPManagement.Entity_Framework;

namespace TRPManagement.Controllers
{
    public class ProgramController : Controller
    {
        TRP_ManagementEntities db = new TRP_ManagementEntities();
        public static Program Convert(ProgramDTO s)
        {
            return new Program()
            {
                ProgramId = s.ProgramId,
                ProgramName = s.ProgramName,
                TRPScore = s.TRPScore,
                ChannelId = s.ChannelId,
                AirTime = s.AirTime,
                Channel = s.Channel,
            };
        }
        public static ProgramDTO Convert(Program s)
        {
            return new ProgramDTO()
            {
                ProgramId = s.ProgramId,
                ProgramName = s.ProgramName,
                TRPScore = s.TRPScore,
                ChannelId = s.ChannelId,
                AirTime = s.AirTime,
                Channel = s.Channel,
            };
        }
        public static List<ProgramDTO> Convert(List<Program> data)
        {
            var list = new List<ProgramDTO>();
            foreach (var s in data)
            {
                list.Add(Convert(s));
            }
            return list;
        }
        public ActionResult ViewProgram()
        {
            var data = db.Programs.ToList();
            return View(Convert(data));
        }
        [HttpGet]
        public ActionResult AddProgram()
        {
            var data = db.Channels.ToList();
            ViewBag.Channels=data;
            return View(new ProgramDTO());
        }
        [HttpPost]
        public ActionResult AddProgram(ProgramDTO s)
        {
            if (ModelState.IsValid)
            {
                db.Programs.Add(Convert(s));
                db.SaveChanges();
                TempData["msg"] = "New Program Successfully Added.";
                return RedirectToAction("ViewProgram");
            }
            var data = db.Channels.ToList();
            ViewBag.Channels = data;
            return View(s);
        }
        [HttpGet]
        public ActionResult EditProgram(int id)
        {
            var channels = db.Channels.ToList();
            ViewBag.Channels = channels;
            var data = db.Programs.Find(id);
            return View(Convert(data));
        }
        [HttpPost]
        public ActionResult EditProgram(ProgramDTO d)
        {
            ModelState.Remove("ProgramName");

            if (ModelState.IsValid)
            {
                var data = db.Programs.Find(d.ProgramId);
                db.Entry(data).CurrentValues.SetValues(Convert(d));
                db.SaveChanges();
                TempData["msg"] = "Program details Updated!";
                return RedirectToAction("ViewProgram");
            }
            var channels = db.Channels.ToList();
            ViewBag.Channels = channels;
            return View(d);
        }
        public ActionResult DeleteProgram(int id)
        {
            var data = db.Programs.Find(id);
            db.Programs.Remove(data);
            db.SaveChanges();
            TempData["msg"] = "Program data Deleted";
            return RedirectToAction("ViewProgram");
        }
        public ActionResult DetailsProgram(int id)
        {
            return View(Convert(db.Programs.Find(id)));
        }
        [HttpGet]
        public ActionResult SearchProgram()
        {
            var programs = db.Programs.ToList();
            return View(programs);
        }
        [HttpPost]
        public ActionResult SearchProgram(string searchBy, string programName, double?minTRP, double?maxTRP)
        {
            var programs = db.Programs.ToList();

            if (!string.IsNullOrEmpty(searchBy))
            {
                switch (searchBy)
                {
                    case "Name":
                        if (!string.IsNullOrEmpty(programName))
                        {
                            programs = (from item in programs
                                       where item.ProgramName.Contains(programName)
                                       select item).ToList();
                        }
                        break;

                    case "TRPRating":
                        if (minTRP.HasValue && maxTRP.HasValue)
                        {
                            programs = (from item in programs
                                        where item.TRPScore >= minTRP.Value &&
                                              item.TRPScore <= maxTRP.Value
                                        select item).ToList();
                        }
                        else if (minTRP.HasValue)
                        {
                            programs = (from item in programs
                                        where item.TRPScore >= minTRP.Value
                                        select item).ToList(); 
                        }
                        break;

                    case "Popularity":
                        programs = (from item in programs
                                    orderby item.TRPScore descending 
                                    select item).ToList();
                        break;
                }
            }

            ViewBag.SearchBy = searchBy;
            ViewBag.programName = programName;
            ViewBag.MinTRP = minTRP;
            ViewBag.MaxTRP = maxTRP;
            return View(programs);
        }

    }
}