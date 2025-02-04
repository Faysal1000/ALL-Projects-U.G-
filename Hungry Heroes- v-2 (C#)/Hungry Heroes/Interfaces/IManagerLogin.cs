using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Hungry_Heroes.Interfaces
{
    internal interface IManagerLogin
    {
        SqlConnection databaseConnection();
        void beyeok_Click(object sender, EventArgs e);
        void beyecross_Click(object sender, EventArgs e);
        void blogin_Click(object sender, EventArgs e);
        void bprevious_Click(object sender, EventArgs e);
        void bexit_Click(object sender, EventArgs e);
        void brecapta_Click(object sender, EventArgs e);

    }
}
