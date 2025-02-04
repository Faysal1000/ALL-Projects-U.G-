using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Hungry_Heroes.Interfaces
{
    internal interface IForgotPasswordHome
    {
        SqlConnection databaseConnection();
        void bprevious_Click(object sender, EventArgs e);
        void bgetpass_Click(object sender, EventArgs e);
        void bexit_Click(object sender, EventArgs e);

    }
}
