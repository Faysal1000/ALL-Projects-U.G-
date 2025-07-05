using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Mvc;

namespace Library_Management.Authorization
{
    public class AdminOnlyAttribute : ActionFilterAttribute
    {
        public override void OnActionExecuting(ActionExecutingContext filterContext)
        {
            var session = filterContext.HttpContext.Session;
            if (session == null || session["UserRole"] == null || session["UserRole"].ToString() != "Admin")
            {
                filterContext.Result = new RedirectResult("/Login/Login");
            }
        }
    }

}