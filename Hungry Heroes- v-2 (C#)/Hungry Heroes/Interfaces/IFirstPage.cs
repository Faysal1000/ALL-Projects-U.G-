using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Hungry_Heroes.Interfaces
{
    internal interface IFirstPage
    {
        void bcustomerlogin_Click(object sender, EventArgs e);
        void bresturantlogin_Click(object sender, EventArgs e);
        void bcustomersignup_Click(object sender, EventArgs e);
        void bresturantsignup_Click(object sender, EventArgs e);
        void bexit_Click(object sender, EventArgs e);
        void badmin_Click(object sender, EventArgs e);
        void bmanager_Click(object sender, EventArgs e);
        void lguest_LinkClicked(object sender, LinkLabelLinkClickedEventArgs e);

    }
}
