using Business_Logic_Layer.DTOs;
using Business_Logic_Layer.Services;
using Contract_Management_System.Authentication;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Web.Http;

namespace Contract_Management_System.Controllers
{
    [Logged]
    public class RegistrationController : ApiController
    {
        [HttpGet]
        [Route("api/User/getall")]
        public HttpResponseMessage Get()
        {
            var data = RegistrationService.Get();
            return Request.CreateResponse(HttpStatusCode.OK, data);
        }
        [HttpGet]
        [Route("api/User/get")]
        public HttpResponseMessage Get(int id)
        {
            var data = RegistrationService.Get(id);
            return Request.CreateResponse(HttpStatusCode.OK, data);
        }
        [HttpPost]
        [Route("api/User/create")]
        public HttpResponseMessage Create(UserDTO obj)
        {
            var data = RegistrationService.Create(obj);
            return Request.CreateResponse(HttpStatusCode.OK, data);
        }
        [HttpGet]
        [Route("api/User/delete")]
        public HttpResponseMessage Delete(int id)
        {
            var data = RegistrationService.Delete(id);
            return Request.CreateResponse(HttpStatusCode.OK, data);
        }
        [HttpPost]
        [Route("api/User/update")]
        public HttpResponseMessage Update(UserDTO obj)
        {
            var data = RegistrationService.Update(obj);
            return Request.CreateResponse(HttpStatusCode.OK, data);
        }
    }
}
