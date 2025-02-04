namespace Hungry_Heroes
{
    partial class First_page
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
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
        ///  Required method for Designer support - do not modify
        ///  the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(First_page));
            ltitle = new Label();
            lnote = new Label();
            badmin = new Button();
            bcustomerlogin = new Button();
            bresturantlogin = new Button();
            bresturantsignup = new Button();
            bcustomersignup = new Button();
            bexit = new Button();
            linstruction = new Label();
            bmanager = new Button();
            lguest = new LinkLabel();
            label1 = new Label();
            SuspendLayout();
            // 
            // ltitle
            // 
            ltitle.Anchor = AnchorStyles.None;
            ltitle.AutoSize = true;
            ltitle.BackColor = Color.Transparent;
            ltitle.Font = new Font("Mistral", 50.25F, FontStyle.Regular, GraphicsUnit.Point);
            ltitle.ForeColor = Color.FromArgb(0, 192, 0);
            ltitle.Location = new Point(228, 9);
            ltitle.Name = "ltitle";
            ltitle.Size = new Size(343, 79);
            ltitle.TabIndex = 0;
            ltitle.Text = "Hungry Heroes";
            // 
            // lnote
            // 
            lnote.Anchor = AnchorStyles.None;
            lnote.AutoSize = true;
            lnote.BackColor = Color.Transparent;
            lnote.Font = new Font("Pristina", 14.25F, FontStyle.Regular, GraphicsUnit.Point);
            lnote.ForeColor = Color.Yellow;
            lnote.Location = new Point(374, 97);
            lnote.Name = "lnote";
            lnote.Size = new Size(242, 27);
            lnote.TabIndex = 1;
            lnote.Text = "-Order Tasty and Fresh Food Anytime";
            // 
            // badmin
            // 
            badmin.Anchor = AnchorStyles.Top | AnchorStyles.Right;
            badmin.BackColor = Color.Brown;
            badmin.Cursor = Cursors.Hand;
            badmin.FlatStyle = FlatStyle.Popup;
            badmin.Image = Properties.Resources.admin1;
            badmin.ImageAlign = ContentAlignment.MiddleLeft;
            badmin.Location = new Point(707, 1);
            badmin.Name = "badmin";
            badmin.Size = new Size(75, 23);
            badmin.TabIndex = 2;
            badmin.Text = "   Admin";
            badmin.UseVisualStyleBackColor = false;
            badmin.Click += badmin_Click;
            // 
            // bcustomerlogin
            // 
            bcustomerlogin.Anchor = AnchorStyles.None;
            bcustomerlogin.BackColor = Color.DarkGray;
            bcustomerlogin.Cursor = Cursors.Hand;
            bcustomerlogin.FlatAppearance.BorderColor = Color.FromArgb(192, 0, 0);
            bcustomerlogin.FlatAppearance.MouseDownBackColor = Color.LightGreen;
            bcustomerlogin.FlatAppearance.MouseOverBackColor = Color.Lime;
            bcustomerlogin.FlatStyle = FlatStyle.Flat;
            bcustomerlogin.Font = new Font("Arial Rounded MT Bold", 9F, FontStyle.Regular, GraphicsUnit.Point);
            bcustomerlogin.Image = Properties.Resources.login;
            bcustomerlogin.ImageAlign = ContentAlignment.MiddleLeft;
            bcustomerlogin.Location = new Point(257, 306);
            bcustomerlogin.Name = "bcustomerlogin";
            bcustomerlogin.Size = new Size(141, 34);
            bcustomerlogin.TabIndex = 3;
            bcustomerlogin.Text = " Customer Login   ";
            bcustomerlogin.TextAlign = ContentAlignment.MiddleRight;
            bcustomerlogin.UseVisualStyleBackColor = false;
            bcustomerlogin.Click += bcustomerlogin_Click;
            // 
            // bresturantlogin
            // 
            bresturantlogin.Anchor = AnchorStyles.None;
            bresturantlogin.BackColor = Color.DarkGray;
            bresturantlogin.Cursor = Cursors.Hand;
            bresturantlogin.FlatAppearance.BorderColor = Color.FromArgb(192, 0, 0);
            bresturantlogin.FlatAppearance.MouseDownBackColor = Color.LightGreen;
            bresturantlogin.FlatAppearance.MouseOverBackColor = Color.Lime;
            bresturantlogin.FlatStyle = FlatStyle.Flat;
            bresturantlogin.Font = new Font("Arial Rounded MT Bold", 9F, FontStyle.Regular, GraphicsUnit.Point);
            bresturantlogin.Image = Properties.Resources.login;
            bresturantlogin.ImageAlign = ContentAlignment.MiddleLeft;
            bresturantlogin.Location = new Point(404, 306);
            bresturantlogin.Name = "bresturantlogin";
            bresturantlogin.Size = new Size(140, 34);
            bresturantlogin.TabIndex = 4;
            bresturantlogin.Text = " Resturant Login   ";
            bresturantlogin.TextAlign = ContentAlignment.MiddleRight;
            bresturantlogin.UseVisualStyleBackColor = false;
            bresturantlogin.Click += bresturantlogin_Click;
            // 
            // bresturantsignup
            // 
            bresturantsignup.Anchor = AnchorStyles.None;
            bresturantsignup.BackColor = Color.DarkGray;
            bresturantsignup.Cursor = Cursors.Hand;
            bresturantsignup.FlatAppearance.BorderColor = Color.FromArgb(192, 0, 0);
            bresturantsignup.FlatAppearance.MouseDownBackColor = Color.DarkCyan;
            bresturantsignup.FlatAppearance.MouseOverBackColor = Color.LightSeaGreen;
            bresturantsignup.FlatStyle = FlatStyle.Flat;
            bresturantsignup.Font = new Font("Arial Rounded MT Bold", 9F, FontStyle.Regular, GraphicsUnit.Point);
            bresturantsignup.Image = (Image)resources.GetObject("bresturantsignup.Image");
            bresturantsignup.ImageAlign = ContentAlignment.MiddleLeft;
            bresturantsignup.Location = new Point(404, 395);
            bresturantsignup.Name = "bresturantsignup";
            bresturantsignup.Size = new Size(140, 34);
            bresturantsignup.TabIndex = 6;
            bresturantsignup.Text = "   Resturant Signup";
            bresturantsignup.TextAlign = ContentAlignment.MiddleRight;
            bresturantsignup.UseVisualStyleBackColor = false;
            bresturantsignup.Click += bresturantsignup_Click;
            // 
            // bcustomersignup
            // 
            bcustomersignup.Anchor = AnchorStyles.None;
            bcustomersignup.BackColor = Color.DarkGray;
            bcustomersignup.Cursor = Cursors.Hand;
            bcustomersignup.FlatAppearance.BorderColor = Color.FromArgb(192, 0, 0);
            bcustomersignup.FlatAppearance.MouseDownBackColor = Color.DarkCyan;
            bcustomersignup.FlatAppearance.MouseOverBackColor = Color.LightSeaGreen;
            bcustomersignup.FlatStyle = FlatStyle.Flat;
            bcustomersignup.Font = new Font("Arial Rounded MT Bold", 9F, FontStyle.Regular, GraphicsUnit.Point);
            bcustomersignup.Image = (Image)resources.GetObject("bcustomersignup.Image");
            bcustomersignup.ImageAlign = ContentAlignment.MiddleLeft;
            bcustomersignup.Location = new Point(257, 395);
            bcustomersignup.Name = "bcustomersignup";
            bcustomersignup.Size = new Size(141, 34);
            bcustomersignup.TabIndex = 5;
            bcustomersignup.Text = "   Customer Signup";
            bcustomersignup.TextAlign = ContentAlignment.MiddleRight;
            bcustomersignup.UseVisualStyleBackColor = false;
            bcustomersignup.Click += bcustomersignup_Click;
            // 
            // bexit
            // 
            bexit.Anchor = AnchorStyles.None;
            bexit.BackColor = Color.DarkGray;
            bexit.Cursor = Cursors.Hand;
            bexit.FlatAppearance.BorderColor = Color.FromArgb(192, 0, 0);
            bexit.FlatAppearance.MouseDownBackColor = Color.DarkRed;
            bexit.FlatAppearance.MouseOverBackColor = Color.Red;
            bexit.FlatStyle = FlatStyle.Flat;
            bexit.Font = new Font("Arial Rounded MT Bold", 9F, FontStyle.Regular, GraphicsUnit.Point);
            bexit.Image = (Image)resources.GetObject("bexit.Image");
            bexit.ImageAlign = ContentAlignment.MiddleLeft;
            bexit.Location = new Point(257, 437);
            bexit.Name = "bexit";
            bexit.Size = new Size(287, 34);
            bexit.TabIndex = 7;
            bexit.Text = "   Exit";
            bexit.UseVisualStyleBackColor = false;
            bexit.Click += bexit_Click;
            // 
            // linstruction
            // 
            linstruction.Anchor = AnchorStyles.None;
            linstruction.AutoSize = true;
            linstruction.BackColor = Color.Transparent;
            linstruction.Font = new Font("Berlin Sans FB", 12F, FontStyle.Regular, GraphicsUnit.Point);
            linstruction.ForeColor = Color.Pink;
            linstruction.Location = new Point(315, 231);
            linstruction.Name = "linstruction";
            linstruction.Size = new Size(179, 18);
            linstruction.TabIndex = 8;
            linstruction.Text = "Please log in to continue.....";
            // 
            // bmanager
            // 
            bmanager.Anchor = AnchorStyles.None;
            bmanager.BackColor = Color.DarkGray;
            bmanager.Cursor = Cursors.Hand;
            bmanager.FlatAppearance.BorderColor = Color.FromArgb(192, 0, 0);
            bmanager.FlatAppearance.MouseDownBackColor = Color.DarkCyan;
            bmanager.FlatAppearance.MouseOverBackColor = Color.SpringGreen;
            bmanager.FlatStyle = FlatStyle.Flat;
            bmanager.Font = new Font("Arial Rounded MT Bold", 9F, FontStyle.Regular, GraphicsUnit.Point);
            bmanager.Image = (Image)resources.GetObject("bmanager.Image");
            bmanager.ImageAlign = ContentAlignment.MiddleLeft;
            bmanager.Location = new Point(257, 351);
            bmanager.Name = "bmanager";
            bmanager.Size = new Size(287, 34);
            bmanager.TabIndex = 10;
            bmanager.Text = "Manager Login";
            bmanager.UseVisualStyleBackColor = false;
            bmanager.Click += bmanager_Click;
            // 
            // lguest
            // 
            lguest.AutoSize = true;
            lguest.BackColor = Color.Transparent;
            lguest.Font = new Font("Britannic Bold", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            lguest.LinkColor = Color.FromArgb(255, 128, 255);
            lguest.Location = new Point(326, 268);
            lguest.Name = "lguest";
            lguest.Size = new Size(144, 16);
            lguest.TabIndex = 11;
            lguest.TabStop = true;
            lguest.Text = "Continue as a Guest?";
            lguest.LinkClicked += lguest_LinkClicked;
            // 
            // label1
            // 
            label1.Anchor = AnchorStyles.None;
            label1.AutoSize = true;
            label1.BackColor = Color.Transparent;
            label1.Font = new Font("Berlin Sans FB", 12F, FontStyle.Regular, GraphicsUnit.Point);
            label1.ForeColor = Color.Pink;
            label1.Location = new Point(384, 251);
            label1.Name = "label1";
            label1.Size = new Size(25, 18);
            label1.TabIndex = 12;
            label1.Text = "Or";
            // 
            // First_page
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            BackgroundImage = Properties.Resources.mainBackground;
            BackgroundImageLayout = ImageLayout.Stretch;
            ClientSize = new Size(784, 511);
            Controls.Add(label1);
            Controls.Add(lguest);
            Controls.Add(bmanager);
            Controls.Add(linstruction);
            Controls.Add(bexit);
            Controls.Add(bresturantsignup);
            Controls.Add(bcustomersignup);
            Controls.Add(bresturantlogin);
            Controls.Add(bcustomerlogin);
            Controls.Add(badmin);
            Controls.Add(lnote);
            Controls.Add(ltitle);
            Icon = (Icon)resources.GetObject("$this.Icon");
            Name = "First_page";
            StartPosition = FormStartPosition.CenterScreen;
            Text = "Hungry Heroes";
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private Label ltitle;
        private Label lnote;
        private Button badmin;
        private Button bcustomerlogin;
        private Button bresturantlogin;
        private Button bresturantsignup;
        private Button bcustomersignup;
        private Button bexit;
        private Label linstruction;
        private Button bmanager;
        private LinkLabel lguest;
        private Label label1;
    }
}