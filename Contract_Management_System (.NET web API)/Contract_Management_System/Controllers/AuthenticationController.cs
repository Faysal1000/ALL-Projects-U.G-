using Business_Logic_Layer.DTOs;
using Business_Logic_Layer.Services;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Web.ApplicationServices;
using System.Web.Http;

namespace Contract_Management_System.Controllers
{
    public class AuthenticationController : ApiController
    {
        [HttpPost]
        [Route("api/login")]
        public HttpResponseMessage Login(UserDTO user)
        {
            if(user.UserName==null || user.Password == null)
            {
                return Request.CreateResponse(HttpStatusCode.BadRequest);
            }
            var data = AuthService.Authenticate(user.UserName, user.Password);
            if (data != null)
            {
                return Request.CreateResponse(HttpStatusCode.OK, data);
            }
            return Request.CreateResponse(HttpStatusCode.NotFound);
        }
        [HttpPost]
        [Route("api/logout")]
        public HttpResponseMessage Logout(TokenDTO token)
        {
            if (token == null)
            {
                return Request.CreateResponse(HttpStatusCode.BadRequest);
            }
            var data = AuthService.Logout(token.Key);
            if (data)
            {
                return Request.CreateResponse(HttpStatusCode.OK, data);
            }
            return Request.CreateResponse(HttpStatusCode.NotFound);
        }
    }
}
