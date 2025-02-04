using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Hungry_Heroes.Interfaces
{
    internal interface IResturantHome
    {
        SqlConnection databaseConnection();
        void bhome_Click(object sender, EventArgs e);
        void bmenu_Click(object sender, EventArgs e);
        void baddmenu_Click(object sender, EventArgs e);
        void bprofits_Click(object sender, EventArgs e);
        void blogout_Click(object sender, EventArgs e);
        void bexit_Click(object sender, EventArgs e);
        void bdelete_Click(object sender, EventArgs e);
        void badd_Click(object sender, EventArgs e);
        void bupdate_Click(object sender, EventArgs e);
        void MenuLoad();
        void ResturantHome_Load(object sender, EventArgs e);
        void Cell_Click(object sender, DataGridViewCellEventArgs e);
        void bclear_Click(object sender, EventArgs e);
        void bstatus_Click(object sender, EventArgs e);
        void bdiscount_Click(object sender, EventArgs e);
        void bupdatediscount_Click(object sender, EventArgs e);


    }
}
