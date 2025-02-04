namespace Hungry_Heroes
{
    partial class GuestHomePage
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(GuestHomePage));
            whitemode = new RadioButton();
            darkmode = new RadioButton();
            lhh = new Label();
            bexit = new Button();
            pb6 = new Panel();
            pb5 = new Panel();
            pb2 = new Panel();
            pb1 = new Panel();
            bfres = new Button();
            bapplytheme = new Button();
            ltheme = new Label();
            lTitles = new Label();
            ptop = new Panel();
            baboutus = new Button();
            bthemes = new Button();
            randomcolor = new RadioButton();
            buttonpanel = new Panel();
            bprevious = new Button();
            bhome = new Button();
            pHome = new Panel();
            ptop.SuspendLayout();
            buttonpanel.SuspendLayout();
            pHome.SuspendLayout();
            SuspendLayout();
            // 
            // whitemode
            // 
            whitemode.Anchor = AnchorStyles.None;
            whitemode.AutoSize = true;
            whitemode.BackColor = Color.Transparent;
            whitemode.FlatAppearance.CheckedBackColor = Color.White;
            whitemode.Font = new Font("Arial Rounded MT Bold", 12F, FontStyle.Regular, GraphicsUnit.Point);
            whitemode.Location = new Point(414, 194);
            whitemode.Name = "whitemode";
            whitemode.Size = new Size(118, 22);
            whitemode.TabIndex = 2;
            whitemode.Text = "White Mode";
            whitemode.UseVisualStyleBackColor = false;
            whitemode.Visible = false;
            // 
            // darkmode
            // 
            darkmode.Anchor = AnchorStyles.None;
            darkmode.AutoSize = true;
            darkmode.BackColor = Color.Transparent;
            darkmode.Checked = true;
            darkmode.FlatAppearance.CheckedBackColor = Color.Black;
            darkmode.Font = new Font("Arial Rounded MT Bold", 12F, FontStyle.Regular, GraphicsUnit.Point);
            darkmode.Location = new Point(237, 194);
            darkmode.Name = "darkmode";
            darkmode.Size = new Size(111, 22);
            darkmode.TabIndex = 0;
            darkmode.TabStop = true;
            darkmode.Text = "Dark Mode";
            darkmode.UseVisualStyleBackColor = false;
            darkmode.Visible = false;
            // 
            // lhh
            // 
            lhh.AutoSize = true;
            lhh.BackColor = Color.Transparent;
            lhh.Font = new Font("Mistral", 14.25F, FontStyle.Regular, GraphicsUnit.Point);
            lhh.ForeColor = Color.White;
            lhh.Location = new Point(22, 27);
            lhh.Name = "lhh";
            lhh.Size = new Size(100, 22);
            lhh.TabIndex = 12;
            lhh.Text = "Hungry Heroes";
            // 
            // bexit
            // 
            bexit.Anchor = AnchorStyles.Left;
            bexit.BackColor = Color.Black;
            bexit.Cursor = Cursors.Hand;
            bexit.FlatAppearance.BorderSize = 0;
            bexit.FlatStyle = FlatStyle.Flat;
            bexit.Font = new Font("Arial Rounded MT Bold", 9F, FontStyle.Regular, GraphicsUnit.Point);
            bexit.ForeColor = Color.White;
            bexit.Location = new Point(2, 317);
            bexit.Name = "bexit";
            bexit.Size = new Size(139, 42);
            bexit.TabIndex = 11;
            bexit.Text = "Exit";
            bexit.UseVisualStyleBackColor = false;
            bexit.Click += bexit_Click;
            // 
            // pb6
            // 
            pb6.Anchor = AnchorStyles.Left;
            pb6.BackColor = Color.Cyan;
            pb6.Location = new Point(2, 262);
            pb6.Name = "pb6";
            pb6.Size = new Size(139, 5);
            pb6.TabIndex = 10;
            pb6.Visible = false;
            // 
            // pb5
            // 
            pb5.Anchor = AnchorStyles.Left;
            pb5.BackColor = Color.Cyan;
            pb5.Location = new Point(3, 216);
            pb5.Name = "pb5";
            pb5.Size = new Size(139, 5);
            pb5.TabIndex = 9;
            pb5.Visible = false;
            // 
            // pb2
            // 
            pb2.Anchor = AnchorStyles.Left;
            pb2.BackColor = Color.Cyan;
            pb2.Location = new Point(3, 175);
            pb2.Name = "pb2";
            pb2.Size = new Size(139, 5);
            pb2.TabIndex = 6;
            pb2.Visible = false;
            // 
            // pb1
            // 
            pb1.Anchor = AnchorStyles.Left;
            pb1.BackColor = Color.Cyan;
            pb1.Location = new Point(3, 129);
            pb1.Name = "pb1";
            pb1.Size = new Size(139, 5);
            pb1.TabIndex = 3;
            pb1.Visible = false;
            // 
            // bfres
            // 
            bfres.Anchor = AnchorStyles.Left;
            bfres.BackColor = Color.Black;
            bfres.Cursor = Cursors.Hand;
            bfres.FlatAppearance.BorderSize = 0;
            bfres.FlatStyle = FlatStyle.Flat;
            bfres.Font = new Font("Arial Rounded MT Bold", 9F, FontStyle.Regular, GraphicsUnit.Point);
            bfres.ForeColor = Color.White;
            bfres.Location = new Point(3, 135);
            bfres.Name = "bfres";
            bfres.Size = new Size(139, 42);
            bfres.TabIndex = 5;
            bfres.Text = "Find Resturant";
            bfres.UseVisualStyleBackColor = false;
            bfres.Click += bfres_Click;
            // 
            // bapplytheme
            // 
            bapplytheme.Anchor = AnchorStyles.None;
            bapplytheme.BackColor = Color.Transparent;
            bapplytheme.Cursor = Cursors.Hand;
            bapplytheme.FlatAppearance.BorderSize = 3;
            bapplytheme.FlatAppearance.MouseDownBackColor = Color.Gray;
            bapplytheme.FlatAppearance.MouseOverBackColor = Color.FromArgb(64, 64, 64);
            bapplytheme.FlatStyle = FlatStyle.Popup;
            bapplytheme.Font = new Font("Arial Rounded MT Bold", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            bapplytheme.ForeColor = Color.Black;
            bapplytheme.Location = new Point(306, 307);
            bapplytheme.Name = "bapplytheme";
            bapplytheme.Size = new Size(139, 42);
            bapplytheme.TabIndex = 13;
            bapplytheme.Text = "Apply Theme";
            bapplytheme.UseVisualStyleBackColor = false;
            bapplytheme.Visible = false;
            bapplytheme.Click += bapplytheme_Click;
            // 
            // ltheme
            // 
            ltheme.Anchor = AnchorStyles.None;
            ltheme.AutoSize = true;
            ltheme.Font = new Font("Arial Rounded MT Bold", 14.25F, FontStyle.Regular, GraphicsUnit.Point);
            ltheme.Location = new Point(190, 121);
            ltheme.Name = "ltheme";
            ltheme.Size = new Size(400, 22);
            ltheme.TabIndex = 3;
            ltheme.Text = "You Can Choose any one of these options...";
            ltheme.Visible = false;
            // 
            // lTitles
            // 
            lTitles.Anchor = AnchorStyles.None;
            lTitles.AutoSize = true;
            lTitles.Font = new Font("Arial Rounded MT Bold", 18F, FontStyle.Regular, GraphicsUnit.Point);
            lTitles.ForeColor = Color.White;
            lTitles.Location = new Point(286, 27);
            lTitles.Name = "lTitles";
            lTitles.Size = new Size(0, 28);
            lTitles.TabIndex = 0;
            // 
            // ptop
            // 
            ptop.BackColor = Color.Black;
            ptop.BackgroundImageLayout = ImageLayout.Stretch;
            ptop.Controls.Add(lTitles);
            ptop.Dock = DockStyle.Top;
            ptop.Location = new Point(139, 0);
            ptop.Name = "ptop";
            ptop.Size = new Size(745, 85);
            ptop.TabIndex = 4;
            // 
            // baboutus
            // 
            baboutus.Anchor = AnchorStyles.Left;
            baboutus.BackColor = Color.Black;
            baboutus.Cursor = Cursors.Hand;
            baboutus.FlatAppearance.BorderSize = 0;
            baboutus.FlatStyle = FlatStyle.Flat;
            baboutus.Font = new Font("Arial Rounded MT Bold", 9F, FontStyle.Regular, GraphicsUnit.Point);
            baboutus.ForeColor = Color.White;
            baboutus.Location = new Point(2, 223);
            baboutus.Name = "baboutus";
            baboutus.Size = new Size(139, 42);
            baboutus.TabIndex = 4;
            baboutus.Text = "About Us";
            baboutus.UseVisualStyleBackColor = false;
            baboutus.Click += baboutus_Click;
            // 
            // bthemes
            // 
            bthemes.Anchor = AnchorStyles.Left;
            bthemes.BackColor = Color.Black;
            bthemes.Cursor = Cursors.Hand;
            bthemes.FlatAppearance.BorderSize = 0;
            bthemes.FlatStyle = FlatStyle.Flat;
            bthemes.Font = new Font("Arial Rounded MT Bold", 9F, FontStyle.Regular, GraphicsUnit.Point);
            bthemes.ForeColor = Color.White;
            bthemes.Location = new Point(3, 178);
            bthemes.Name = "bthemes";
            bthemes.Size = new Size(139, 42);
            bthemes.TabIndex = 3;
            bthemes.Text = "Themes";
            bthemes.UseVisualStyleBackColor = false;
            bthemes.Click += bthemes_Click;
            // 
            // randomcolor
            // 
            randomcolor.Anchor = AnchorStyles.None;
            randomcolor.AutoSize = true;
            randomcolor.BackColor = Color.Transparent;
            randomcolor.FlatAppearance.CheckedBackColor = Color.Chartreuse;
            randomcolor.Font = new Font("Arial Rounded MT Bold", 12F, FontStyle.Regular, GraphicsUnit.Point);
            randomcolor.Location = new Point(306, 250);
            randomcolor.Name = "randomcolor";
            randomcolor.Size = new Size(139, 22);
            randomcolor.TabIndex = 1;
            randomcolor.Text = "Random Color";
            randomcolor.UseVisualStyleBackColor = false;
            randomcolor.Visible = false;
            // 
            // buttonpanel
            // 
            buttonpanel.BackColor = Color.Black;
            buttonpanel.BackgroundImageLayout = ImageLayout.Stretch;
            buttonpanel.Controls.Add(bprevious);
            buttonpanel.Controls.Add(lhh);
            buttonpanel.Controls.Add(bexit);
            buttonpanel.Controls.Add(pb6);
            buttonpanel.Controls.Add(pb5);
            buttonpanel.Controls.Add(pb2);
            buttonpanel.Controls.Add(pb1);
            buttonpanel.Controls.Add(bfres);
            buttonpanel.Controls.Add(baboutus);
            buttonpanel.Controls.Add(bthemes);
            buttonpanel.Controls.Add(bhome);
            buttonpanel.Dock = DockStyle.Left;
            buttonpanel.Location = new Point(0, 0);
            buttonpanel.Name = "buttonpanel";
            buttonpanel.Size = new Size(139, 561);
            buttonpanel.TabIndex = 3;
            // 
            // bprevious
            // 
            bprevious.Anchor = AnchorStyles.Left;
            bprevious.BackColor = Color.Black;
            bprevious.Cursor = Cursors.Hand;
            bprevious.FlatAppearance.BorderSize = 0;
            bprevious.FlatStyle = FlatStyle.Flat;
            bprevious.Font = new Font("Arial Rounded MT Bold", 9F, FontStyle.Regular, GraphicsUnit.Point);
            bprevious.ForeColor = Color.White;
            bprevious.Location = new Point(2, 269);
            bprevious.Name = "bprevious";
            bprevious.Size = new Size(139, 42);
            bprevious.TabIndex = 13;
            bprevious.Text = "Previous";
            bprevious.UseVisualStyleBackColor = false;
            bprevious.Click += bprevious_Click;
            // 
            // bhome
            // 
            bhome.Anchor = AnchorStyles.Left;
            bhome.BackColor = Color.Black;
            bhome.Cursor = Cursors.Hand;
            bhome.FlatAppearance.BorderSize = 0;
            bhome.FlatStyle = FlatStyle.Flat;
            bhome.Font = new Font("Arial Rounded MT Bold", 9F, FontStyle.Regular, GraphicsUnit.Point);
            bhome.ForeColor = Color.White;
            bhome.Location = new Point(3, 91);
            bhome.Name = "bhome";
            bhome.Size = new Size(139, 42);
            bhome.TabIndex = 0;
            bhome.Text = "Home";
            bhome.UseVisualStyleBackColor = false;
            bhome.Click += bhome_Click;
            // 
            // pHome
            // 
            pHome.Anchor = AnchorStyles.None;
            pHome.BackColor = Color.DarkGray;
            pHome.Controls.Add(bapplytheme);
            pHome.Controls.Add(ltheme);
            pHome.Controls.Add(whitemode);
            pHome.Controls.Add(randomcolor);
            pHome.Controls.Add(darkmode);
            pHome.Location = new Point(139, 85);
            pHome.Name = "pHome";
            pHome.Size = new Size(745, 476);
            pHome.TabIndex = 5;
            // 
            // GuestHomePage
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(884, 561);
            Controls.Add(ptop);
            Controls.Add(buttonpanel);
            Controls.Add(pHome);
            Icon = (Icon)resources.GetObject("$this.Icon");
            Name = "GuestHomePage";
            StartPosition = FormStartPosition.CenterScreen;
            Text = "GuestHomePage";
            ptop.ResumeLayout(false);
            ptop.PerformLayout();
            buttonpanel.ResumeLayout(false);
            buttonpanel.PerformLayout();
            pHome.ResumeLayout(false);
            pHome.PerformLayout();
            ResumeLayout(false);
        }

        #endregion

        private RadioButton whitemode;
        private RadioButton darkmode;
        private Label lhh;
        private Button bexit;
        private Panel pb6;
        private Panel pb5;
        private Panel pb2;
        private Panel pb1;
        private Button bfres;
        private Button bapplytheme;
        private Label ltheme;
        private Label lTitles;
        private Panel ptop;
        private Button baboutus;
        private Button bthemes;
        private RadioButton randomcolor;
        private Panel buttonpanel;
        private Button bhome;
        private Panel pHome;
        private Button bprevious;
    }
}