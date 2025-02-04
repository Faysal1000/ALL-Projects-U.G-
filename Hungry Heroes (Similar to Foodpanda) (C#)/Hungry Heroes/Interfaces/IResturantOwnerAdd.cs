using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Hungry_Heroes.Interfaces
{
    internal interface IResturantOwnerAdd
    {
        SqlConnection databaseConnection();
        void updateDataLoad();
        void ResturantOwnerAdd_Load(object sender, EventArgs e);
        void badd_Click(object sender, EventArgs e);
        void bupdate_Click(object sender, EventArgs e);
        void bcancel_Click(object sender, EventArgs e);
        bool IsValidEmail(string emailAddress);

    }
}
