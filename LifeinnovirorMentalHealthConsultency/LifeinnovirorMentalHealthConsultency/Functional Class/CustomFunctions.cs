using System;
using System.Linq;
using System.Security.Claims;
using System.Security.Cryptography;
using System.Security.Principal;
using System.Text;
using LifeinnovirorMentalHealthConsultency.Context;

public static class CustomFunctions
{
    private static LifeinnovirorContext db = new LifeinnovirorContext();
    
    //this will return current logged in admin user id
    public static int GetAdminUserIdFromToken(IPrincipal user)
    {
        var identity = user.Identity as ClaimsIdentity;
        var email = identity?.FindFirst(ClaimTypes.Email)?.Value;

        if (string.IsNullOrEmpty(email))
            return 0;

        var admin = db.Admins.FirstOrDefault(a => a.Email == email);
        return admin?.AdminId ?? 0;
    }

    //this will return current logged in doctor user id
    public static int GetDoctorUserIdFromToken(IPrincipal user)
    {
        var identity = user.Identity as ClaimsIdentity;
        var email = identity?.FindFirst(ClaimTypes.Email)?.Value;

        if (string.IsNullOrEmpty(email))
            return 0;

        var doctor = db.Doctors.FirstOrDefault(a => a.Email == email);
        return doctor?.DoctorId ?? 0;
    }

    //this will return current logged in patient user id
    public static int GetPatientUserIdFromToken(IPrincipal user)
    {
        var identity = user.Identity as ClaimsIdentity;
        var email = identity?.FindFirst(ClaimTypes.Email)?.Value;

        if (string.IsNullOrEmpty(email))
            return 0;

        var patient = db.Patients.FirstOrDefault(a => a.Email == email);
        return patient?.PatientId ?? 0;
    }


    // this is a function which will be used to secure password using hashing
    public static string GetSha256HashBase64(string input)
    {
        using (var sha256 = SHA256.Create())
        {
            var inputBytes = Encoding.UTF8.GetBytes(input);
            var hashBytes = sha256.ComputeHash(inputBytes);
            return Convert.ToBase64String(hashBytes);
        }
    }
}
