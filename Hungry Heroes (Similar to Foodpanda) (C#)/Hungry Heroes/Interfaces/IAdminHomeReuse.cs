using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Hungry_Heroes.Interfaces
{
    internal interface IAdminHomeReuse
    {
        SqlConnection databaseConnection();
        void AdminHomeReuse_Load(object sender, EventArgs e);

    }
}
