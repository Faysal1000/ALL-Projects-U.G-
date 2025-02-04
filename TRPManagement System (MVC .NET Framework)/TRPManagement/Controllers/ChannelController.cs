using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Mvc;
using TRPManagement.DTOs;
using TRPManagement.Entity_Framework;

namespace TRPManagement.Controllers
{
    public class ChannelController : Controller
    {

        TRP_ManagementEntities db = new TRP_ManagementEntities();
        public static Channel Convert(ChannelDTO s)
        {
            return new Channel()
            {
                ChannelId = s.ChannelId,
                ChannelName = s.ChannelName,
                EstablishedYear = s.EstablishedYear,
                Country = s.Country,
                Programs = s.Programs
            };
        }
        public static ChannelDTO Convert(Channel s)
        {
            return new ChannelDTO()
            {
                ChannelId = s.ChannelId,
                ChannelName = s.ChannelName,
                EstablishedYear = s.EstablishedYear,
                Country = s.Country,
                Programs = (List<Program>)s.Programs
            };
        }
        public static List<ChannelDTO> Convert(List<Channel> data)
        {
            var list = new List<ChannelDTO>();
            foreach (var s in data)
            {
                list.Add(Convert(s));
            }
            return list;
        }
        public ActionResult ViewChannel()
        {
            var data = db.Channels.ToList();
            return View(Convert(data));
        }
        [HttpGet]
        public ActionResult AddChannel()
        {
            return View(new ChannelDTO());
        }
        [HttpPost]
        public ActionResult AddChannel(ChannelDTO s)
        {
            if (ModelState.IsValid)
            {
                db.Channels.Add(Convert(s));
                db.SaveChanges();
                TempData["msg"] = "New Channel Added Successfully";
                return RedirectToAction("ViewChannel");
            }
            return View(s);
        }
        [HttpGet]
        public ActionResult EditChannel(int id)
        {
            var data = db.Channels.Find(id);
            return View(Convert(data));
        }
        [HttpPost]
        public ActionResult EditChannel(ChannelDTO d)
        {
            ModelState.Remove("ChannelName");
            if (ModelState.IsValid)
            {
                var data = db.Channels.Find(d.ChannelId);
                db.Entry(data).CurrentValues.SetValues(Convert(d));
                db.SaveChanges();
                TempData["msg"] = "Channels Detail Updated";
                return RedirectToAction("ViewChannel");
            }
            return View(d);
        }
        public ActionResult DeleteChannel(int id)
        {
            var data = db.Channels.Find(id);
            if (data.Programs.Any())
            {
                TempData["msg"] = "This channel has associated programs. Can't delete!";
                return RedirectToAction("ViewChannel");
            }
            db.Channels.Remove(data);
            db.SaveChanges();
            TempData["msg"] = "Channel data Deleted";
            return RedirectToAction("ViewChannel");
        }
        public ActionResult DetailsChannel(int id)
        {
            return View(Convert(db.Channels.Find(id)));
        }

    }
}