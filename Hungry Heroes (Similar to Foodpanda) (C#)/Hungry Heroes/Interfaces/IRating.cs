using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Hungry_Heroes.Interfaces
{
    internal interface IRating
    {
        SqlConnection databaseConnection();
        void brate1_Click(object sender, EventArgs e);
        void rate2_Click(object sender, EventArgs e);
        void button2_Click(object sender, EventArgs e);
        void button3_Click(object sender, EventArgs e);
        void button4_Click(object sender, EventArgs e);
        void bprevious_Click(object sender, EventArgs e);

    }
}
