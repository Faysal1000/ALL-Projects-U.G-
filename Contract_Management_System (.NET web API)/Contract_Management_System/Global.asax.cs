using Business_Logic_Layer.Services;
using Contract_Management_System.Controllers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Web;
using System.Web.Http;
using System.Web.Routing;

namespace Contract_Management_System
{
    public class WebApiApplication : System.Web.HttpApplication
    {
        protected void Application_Start()
        {
            GlobalConfiguration.Configure(WebApiConfig.Register);

            //ContactService.SendEmailForBirthdayReminder();
            Console.WriteLine("Function executed at: " + DateTime.Now);

            Timer timer = null;
            var interval = TimeSpan.FromHours(24);

            timer = new Timer(_ =>
            {
                ContactService.SendEmailForBirthdayReminder();
                Console.WriteLine("Function executed at: " + DateTime.Now);
            }, null, interval, interval);
        }
    }
}
