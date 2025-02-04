using Data_Access_Layer.Entity_Framework.Tables;
using Data_Access_Layer.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Data_Access_Layer.Repos
{
    internal class TokenRepo:Repo, IRepo<Token, int, bool>, IGetTokenByKey
    {
        public bool Create(Token obj)
        {
            db.Tokens.Add(obj);
            return db.SaveChanges() > 0;
        }
        public Token Get(int id)
        {
            return db.Tokens.Find(id);
        }

        public List<Token> Get()
        {
            return db.Tokens.ToList();
        }
        public bool Delete(int id)
        {
            return false;
        }
        public bool Update(Token obj)
        {
            var exob = Get(obj.Id);
            db.Entry(exob).CurrentValues.SetValues(obj);
            return db.SaveChanges() > 0;
        }

        public Token Get(string key)
        {
            return (from token in db.Tokens
                    where  token.Key == key
                    select token).SingleOrDefault();
        }
    }
}
