using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Hungry_Heroes.Interfaces
{
    internal interface IChoosePaymentMethod
    {
        SqlConnection databaseConnection();
        void bbkash_Click(object sender, EventArgs e);
        void bmastercard_Click(object sender, EventArgs e);
        void bclose_Click(object sender, EventArgs e);
        void CheckOutDisplayLoad();
        void ChoosePaymentMethod_Load(object sender, EventArgs e);
        void button1_Click(object sender, EventArgs e);

    }
}
