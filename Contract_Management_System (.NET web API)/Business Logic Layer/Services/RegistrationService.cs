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
    public class RegistrationService
    {
        public static Mapper GetMapper()
        {
            var config = new MapperConfiguration(cfg => {
                cfg.CreateMap<User, UserDTO>();
                cfg.CreateMap<UserDTO, User>();
            });
            return new Mapper(config);
        }
        public static UserDTO Get(int id)
        {
            return GetMapper().Map<UserDTO>(DataManager.UserData().Get(id));
        }
        public static List<UserDTO> Get()
        {
            return GetMapper().Map<List<UserDTO>>(DataManager.UserData().Get());
        }
        public static bool Create(UserDTO obj)
        {
            if (obj.Password != null && obj.UserName!=null)
            {
                obj.Password = CreateMD5(obj.Password);
                return DataManager.UserData().Create(GetMapper().Map<User>(obj));
            }
            return false;
        }
        public static bool Delete(int id)
        {
            return DataManager.UserData().Delete(id);
        }
        public static bool Update(UserDTO obj)
        {
            return DataManager.UserData().Update(GetMapper().Map<User>(obj));
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
