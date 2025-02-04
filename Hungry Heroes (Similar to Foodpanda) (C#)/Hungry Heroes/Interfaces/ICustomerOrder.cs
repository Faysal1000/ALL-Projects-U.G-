using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Hungry_Heroes.Interfaces
{
    internal interface ICustomerOrder
    {
        SqlConnection databaseConnection();
        void baddfood_Click(object sender, EventArgs e);
        void bexit_Click(object sender, EventArgs e);
        void bprevious_Click(object sender, EventArgs e);
        void bcheckout_Click(object sender, EventArgs e);
        void bplus_Click(object sender, EventArgs e);
        void bminus_Click(object sender, EventArgs e);
        void CustomerOrder_Load(object sender, EventArgs e);
        void display_CellClick(object sender, DataGridViewCellEventArgs e);
        bool checkDiscount();

    }
}
