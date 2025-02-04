using Data_Access_Layer.Entity_Framework.Tables;
using Data_Access_Layer.Interfaces;
using Data_Access_Layer.Repos;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Text;
using System.Threading.Tasks;

namespace Data_Access_Layer.Data_Manager
{
    public class DataManager
    {
        public static IRepo<Contact, int, bool> ContactData()
        {
            return new ContactRepo();
        }
        public static IContactFunctionalities ContactFunction()
        {
            return new ContactRepo();
        }
        public static IRepo<User, int, bool> UserData()
        {
            return new UserRepo();
        }
        public static IUserAuthentication AuthData()
        {
            return new UserRepo();
        }
        public static IRepo<Token, int, bool> TokenData()
        {
            return new TokenRepo();
        }
        public static IGetTokenByKey TokenByKey()
        {
            return new TokenRepo();
        }
    }
}
