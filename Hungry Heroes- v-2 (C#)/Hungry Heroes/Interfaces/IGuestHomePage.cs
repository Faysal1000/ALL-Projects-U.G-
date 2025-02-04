using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Hungry_Heroes.Interfaces
{
    internal interface IGuestHomePage
    {
        Color getColor();
        void bhome_Click(object sender, EventArgs e);
        void bfres_Click(object sender, EventArgs e);
        void bporder_Click(object sender, EventArgs e);
        void bthemes_Click(object sender, EventArgs e);
        void baboutus_Click(object sender, EventArgs e);
        void bexit_Click(object sender, EventArgs e);
        void bapplytheme_Click(object sender, EventArgs e);
        void bprevious_Click(object sender, EventArgs e);

    }
}
