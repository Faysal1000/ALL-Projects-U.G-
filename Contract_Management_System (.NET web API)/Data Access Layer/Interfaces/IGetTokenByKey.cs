using Data_Access_Layer.Entity_Framework.Tables;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Data_Access_Layer.Interfaces
{
    public interface IGetTokenByKey
    {
        Token Get(string key);
    }
}
