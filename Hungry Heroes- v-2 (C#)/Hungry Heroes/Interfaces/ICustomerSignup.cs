using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Hungry_Heroes.Interfaces
{
    internal interface ICustomerSignup
    {
        SqlConnection databaseConnection();
        bool IsPasswordValid(string password);
        void bsignup_Click(object sender, EventArgs e);
        void bprevious_Click(object sender, EventArgs e);
        void bexit_Click(object sender, EventArgs e);
        void brecapta_Click(object sender, EventArgs e);
        bool IsValidEmail(string emailAddress);

    }
}
