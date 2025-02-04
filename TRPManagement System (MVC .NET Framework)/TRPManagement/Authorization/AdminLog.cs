using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Mvc;
using TRPManagement.DTOs;

namespace TRPManagement.Authorization
{
    public class AdminLog: AuthorizeAttribute
    {
        protected override bool AuthorizeCore(HttpContextBase httpContext)
        {
            var user = (AdminDTO)httpContext.Session["user"];
            if (user != null)
            {
                return true;
            }
            return false;
        }

    }
}