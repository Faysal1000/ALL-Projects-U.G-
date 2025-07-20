using System;
using System.Configuration;
using System.Text;
using Microsoft.IdentityModel.Tokens;
using Microsoft.Owin.Security.DataHandler.Encoder;
using Microsoft.Owin.Security.Jwt;
using Owin;
using Microsoft.Owin.Security;
using Microsoft.Owin;

[assembly: OwinStartup(typeof(LifeinnovirorMentalHealthConsultency.Startup))]

namespace LifeinnovirorMentalHealthConsultency
{
    public class Startup
    {
        public void Configuration(IAppBuilder app)
        {
            app.UseCors(Microsoft.Owin.Cors.CorsOptions.AllowAll);

            var issuer = ConfigurationManager.AppSettings["JwtIssuer"];
            var audience = ConfigurationManager.AppSettings["JwtAudience"];

            // Decode only once — assuming JwtKey is Base64 encoded 256-bit key
            var secret = TextEncodings.Base64Url.Decode(ConfigurationManager.AppSettings["JwtKey"]);

            app.UseJwtBearerAuthentication(new JwtBearerAuthenticationOptions
            {
                AuthenticationMode = AuthenticationMode.Active,
                TokenValidationParameters = new TokenValidationParameters
                {
                    ValidAudience = audience,
                    ValidIssuer = issuer,
                    IssuerSigningKey = new SymmetricSecurityKey(secret),
                    ValidateIssuerSigningKey = true,
                    ValidateLifetime = true,
                    ClockSkew = TimeSpan.Zero,

                    // Use standard ClaimTypes 
                    NameClaimType = System.Security.Claims.ClaimTypes.Email,
                    RoleClaimType = System.Security.Claims.ClaimTypes.Role,
                }
            });
        }
    }
}
