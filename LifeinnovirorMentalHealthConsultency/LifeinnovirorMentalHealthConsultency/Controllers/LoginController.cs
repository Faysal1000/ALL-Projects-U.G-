using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Security.Claims;
using System.Security.Principal;
using System.Web.Http;
using System.Web.Services.Description;
using LifeinnovirorMentalHealthConsultency.Authorization;
using LifeinnovirorMentalHealthConsultency.Context;
using LifeinnovirorMentalHealthConsultency.Models;
using static LifeinnovirorMentalHealthConsultency.Models.FakeDB;

namespace LifeinnovirorMentalHealthConsultency.Controllers
{
    public class LoginController : ApiController
    {
        private LifeinnovirorContext db; // Database context

        public LoginController()
        {
            db = new LifeinnovirorContext(); // Initialize the DB context
        }

        /// <summary>
        /// Admin login with email and password (hashed with MD5).
        /// Returns JWT token on success.
        /// </summary>
        [HttpPost]
        [Route("api/admin/login")]
        public IHttpActionResult AdminLogin(LoginModel model)
        {
            if (model == null || string.IsNullOrEmpty(model.Email) || string.IsNullOrEmpty(model.Password))
                return Content(HttpStatusCode.BadRequest, 
                    new { message = "Email or password is missing." });

            var passwordHash = Hashing.CreateMD5(model.Password);

            // Check if credentials match
            var user = db.Admins.FirstOrDefault(u => u.Email == model.Email &&
                                                     u.PasswordHash == passwordHash);

            if (user == null)
                return Unauthorized();

            var token = TokenManager.GenerateToken(user.Email, "Admin");

            return Ok(new
            {
                token = token
            });
        }

        /// <summary>
        /// Doctor login with plain password match 
        /// Returns JWT token on success.
        /// </summary>
        [HttpPost]
        [Route("api/doctor/login")]
        public IHttpActionResult DoctorLogin(LoginModel model)
        {
            if (model == null || string.IsNullOrEmpty(model.Email) || string.IsNullOrEmpty(model.Password))
                return Content(HttpStatusCode.BadRequest, 
                    new { message = "Email or password is missing." });

            var passwordHash = Hashing.CreateMD5(model.Password);

            // Check if credentials match
            var user = db.Doctors.FirstOrDefault(u => u.Email == model.Email &&
                                                      u.PasswordHash == passwordHash); 

            if (user == null)
                return Unauthorized();

            var token = TokenManager.GenerateToken(user.Email, "Doctor");

            return Ok(new
            {
                token = token
            });
        }

        /// <summary>
        /// Patient login. Warns if using default password (email as password).
        /// Returns JWT token and optional warning message.
        /// </summary>
        [HttpPost]
        [Route("api/patient/login")]
        public IHttpActionResult PatientLogin(LoginModel model)
        {
            if (model == null || string.IsNullOrEmpty(model.Email) || string.IsNullOrEmpty(model.Password))
                return Content(HttpStatusCode.BadRequest, 
                    new { message = "Email or password is missing." });

            var passwordHash = Hashing.CreateMD5(model.Password);

            // Check if credentials match with hashed
            var user = db.Patients.FirstOrDefault(u => u.Email == model.Email &&
                                                      u.PasswordHash == passwordHash);
            
            if (user == null)
                return Unauthorized();

            // Warn if password is same as email
            string message = "";
            if (Hashing.CreateMD5(user.Email) == user.PasswordHash)
            {
                message = "It is recommended to change your password as your default password is your email address.";
            }

            var token = TokenManager.GenerateToken(user.Email, "Patient");

            return Ok(new
            {
                token = token,
                message = message
            });
        }

        /// <summary>
        /// Test API to verify that role-based authentication is working.
        /// Can be accessed by Admin, Doctor, or Patient roles only.
        /// </summary>
        [Authorize(Roles = "Admin,Doctor,Patient")]
        [HttpGet]
        [Route("api/helloWorld")]
        public IHttpActionResult HelloWorld()
        {
            var identity = User.Identity as ClaimsIdentity;

            var email = identity.FindFirst(ClaimTypes.Email)?.Value ?? "Unknown";
            var role = identity.FindFirst(ClaimTypes.Role)?.Value ?? "No Role";

            return Ok($"Hello {email}, your role is {role} and you are authenticated!");
        }
    }
}
