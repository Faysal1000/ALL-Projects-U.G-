using System;
using System.Configuration;
using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using System.Text;
using Microsoft.IdentityModel.Tokens;

namespace LifeinnovirorMentalHealthConsultency.Authorization
{
    public class TokenManager
    {
        public static string GenerateToken(string userEmail, string role, int expirationInMinutes = 300)
        {
            var secretKey = ConfigurationManager.AppSettings["JwtKey"];
            var issuer = ConfigurationManager.AppSettings["JwtIssuer"];
            var audience = ConfigurationManager.AppSettings["JwtAudience"];

            // Key must match the decoding approach used in Startup.cs
            var key = Convert.FromBase64String(secretKey);

            var claims = new[]
            {
                new Claim(ClaimTypes.Email, userEmail),
                new Claim(ClaimTypes.Role, role)
            };

            var tokenDescriptor = new SecurityTokenDescriptor
            {
                Subject = new ClaimsIdentity(claims),
                Expires = DateTime.UtcNow.AddMinutes(expirationInMinutes),
                SigningCredentials = new SigningCredentials(new SymmetricSecurityKey(key), SecurityAlgorithms.HmacSha256Signature),
                Issuer = issuer,
                Audience = audience
            };

            var tokenHandler = new JwtSecurityTokenHandler();
            var token = tokenHandler.CreateToken(tokenDescriptor);
            return tokenHandler.WriteToken(token);
        }
    }
}
