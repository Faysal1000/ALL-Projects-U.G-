using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;

namespace Hungry_Heroes.Interfaces
{
    public interface IAdminHomePage
    {
        SqlConnection databaseConnection();
        void bhome_Click(object sender, EventArgs e);
        void bresturants_Click(object sender, EventArgs e);
        void bprofile_Click(object sender, EventArgs e);
        void bmanager_Click(object sender, EventArgs e);
        void bguest_Click(object sender, EventArgs e);
        void bprofits_Click(object sender, EventArgs e);
        void blogout_Click(object sender, EventArgs e);
        void bexit_Click(object sender, EventArgs e);
        void badd_Click(object sender, EventArgs e);
        void bdelete_Click(object sender, EventArgs e);
        void bupdate_Click_1(object sender, EventArgs e);
        void showResturnatDetails();
        void showCustomertDetails();
        void showGuestUser();
        void showLoginDetails();
        void showCustomerLoginDetails();
        void showResturantOwnerLoginDetails();
        void showGuestLoginDetails();
        void showManagerLoginDetails();
        void showManagerDetails();
        void blogmanager_Click(object sender, EventArgs e);
    }
}
