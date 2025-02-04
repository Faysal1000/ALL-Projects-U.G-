using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Hungry_Heroes.Interfaces
{
    internal interface IBkashPaymentHome
    {
        SqlConnection databaseConnection();
        void bprevious_Click(object sender, EventArgs e);
        void blogin_Click(object sender, EventArgs e);

    }
}
