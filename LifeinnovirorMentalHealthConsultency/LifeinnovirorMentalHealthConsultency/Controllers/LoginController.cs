using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Security.Claims;
using System.Security.Principal;
using System.Web.Http;
using LifeinnovirorMentalHealthConsultency.Authorization;
using LifeinnovirorMentalHealthConsultency.Models;
using static LifeinnovirorMentalHealthConsultency.Models.FakeDB;

namespace LifeinnovirorMentalHealthConsultency.Controllers
{
    public class LoginController : ApiController
    {
        [HttpPost]
        [Route("api/login")]
        public IHttpActionResult Login(LoginModel model)
        {
            var user = FakeUserStore.FirstOrDefault(u => u.Email == model.Email && u.Password == model.Password);

            if (user == null)
                return Unauthorized();

            var token = TokenManager.GenerateToken(user.Email, user.Role); 

            return Ok(new { token });
        }

        [Authorize (Roles="Admin")]
        [HttpGet]
        [Route("api/helloWorld")]
        public IHttpActionResult HelloWorld()
        {
            var identity = User.Identity as ClaimsIdentity;

            var email = identity.FindFirst(System.Security.Claims.ClaimTypes.Email)?.Value ?? "Unknown";
            var role = identity.FindFirst(System.Security.Claims.ClaimTypes.Role)?.Value ?? "No Role";


            return Ok($"Hello {email}, your role is {role} and you are authenticated!");
        }

    }
}
