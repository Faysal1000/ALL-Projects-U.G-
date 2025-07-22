using System.Linq;
using System.Security.Claims;
using System.Security.Principal;
using System.Text;
using System.Web.Http;
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
    public static string CreateMD5(string input)
    {
        // Use input string to calculate MD5 hash
        using (System.Security.Cryptography.MD5 md5 = System.Security.Cryptography.MD5.Create())
        {
            byte[] inputBytes = System.Text.Encoding.ASCII.GetBytes(input);
            byte[] hashBytes = md5.ComputeHash(inputBytes);

            //return Convert.ToHexString(hashBytes); // .NET 5 +

            //Convert the byte array to hexadecimal string prior to.NET 5
            StringBuilder sb = new System.Text.StringBuilder();
            for (int i = 0; i < hashBytes.Length; i++)
            {
                sb.Append(hashBytes[i].ToString("X2"));
            }
            return sb.ToString();
        }
    }
}
