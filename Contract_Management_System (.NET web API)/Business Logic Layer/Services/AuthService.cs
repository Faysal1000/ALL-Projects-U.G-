using AutoMapper;
using Business_Logic_Layer.DTOs;
using Data_Access_Layer.Data_Manager;
using Data_Access_Layer.Entity_Framework.Tables;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Business_Logic_Layer.Services
{
    public class AuthService
    {
        public static Mapper GetMapper()
        {
            var config = new MapperConfiguration(cfg => {
                cfg.CreateMap<User, UserDTO>();
                cfg.CreateMap<Token, TokenDTO>();
            });
            return new Mapper(config);
        }
        public static TokenDTO Authenticate(string uname, string pass)
        {
            var auth = DataManager.AuthData();
            pass = CreateMD5(pass).ToLower();
            var user = auth.Authenticate(uname, pass);
            if (user != null)
            {
                var token = new Token();
                token.Key = Guid.NewGuid().ToString();
                token.CreatedAt = DateTime.Now;
                token.UserId = user.Id;
                DataManager.TokenData().Create(token);
                return GetMapper().Map<TokenDTO>(DataManager.TokenByKey().Get(token.Key));
            }
            return null;
        }
        public static bool IsTokenValid(string key)
        {
            var tk = GetMapper().Map<TokenDTO>(DataManager.TokenByKey().Get(key));
            if (tk != null && tk.ExpiredAt == null)
            {
                return true;
            }
            return false;
        }
        public static bool Logout(string key)
        {
            var tk = DataManager.TokenByKey().Get(key);
            tk.ExpiredAt = DateTime.Now;
            return DataManager.TokenData().Update(tk);

        }
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
}
