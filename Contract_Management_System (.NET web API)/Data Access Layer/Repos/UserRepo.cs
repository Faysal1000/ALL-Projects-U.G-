using Data_Access_Layer.Entity_Framework.Tables;
using Data_Access_Layer.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Data_Access_Layer.Repos
{
    internal class UserRepo : Repo, IRepo<User, int, bool>, IUserAuthentication
    {
        public bool Create(User obj)
        {
            db.Users.Add(obj);
            return db.SaveChanges() > 0;
        }
        public User Get(int id)
        {
            return db.Users.Find(id);
        }

        public List<User> Get()
        {
            return db.Users.ToList();
        }
        public bool Delete(int id)
        {
            var exob = Get(id);
            db.Users.Remove(exob);
            return db.SaveChanges() > 0;
        }
        public bool Update(User obj)
        {
            var exob = Get(obj.Id);
            db.Entry(exob).CurrentValues.SetValues(obj);
            return db.SaveChanges() > 0;
        }

        public User Authenticate(string uname, string pass)
        {
            return db.Users.Where(
                            u => u.UserName.Equals(uname) &&
                            u.Password.Equals(pass)).SingleOrDefault();
        }
    }
}
