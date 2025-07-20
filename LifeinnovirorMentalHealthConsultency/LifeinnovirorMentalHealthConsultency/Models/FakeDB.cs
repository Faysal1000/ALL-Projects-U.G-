using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;

namespace LifeinnovirorMentalHealthConsultency.Models
{
    public class FakeDB
    {
        public static class FakeUserStore
        {
            public static List<User> Users = new List<User>
            {
                new User { Email = "admin", Password = "123", Role = "Admin" },
                new User { Email = "doctor", Password = "123", Role = "Clinician" },
                new User { Email = "patient", Password = "123", Role = "Patient" }
            };

            public static User FirstOrDefault(Func<User, bool> predicate) => Users.FirstOrDefault(predicate);
        }

        public class User
        {
            public string Email { get; set; }
            public string Password { get; set; }
            public string Role { get; set; }
        }

    }
}