using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Mvc;
using TRPManagement.DTOs;

namespace TRPManagement.Controllers
{
    public class AdminController : Controller
    {
        [HttpGet]
        public ActionResult AdminLogin()
        {
            return View();
        }
        [HttpPost]
        public ActionResult AdminLogin(AdminDTO a)
        {
            /*
            //if db is used then this code will be needed
            var user = (from u in db.Users
                        where u.UName.Equals(log.UName) &&
                        u.Password.Equals(log.Password)
                        select u).SingleOrDefault();
            if (user != null) {
                Session["user"] = user; //boxing
                return RedirectToAction("List","Department");
            }
            TempData["Msg"] = "User not found";
            return View(); 
             */

            if (ModelState.IsValid) {
                if(a.username=="admin" && a.password == "admin")
                {
                    Session["user"] = a;
                    return RedirectToAction("ViewProgram", "Program");
                }
                else
                {
                    TempData["msg"] = "Username or Password incorrect";
                }
            }
            return View(a);
        }
    }
}