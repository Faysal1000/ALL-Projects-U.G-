namespace Hungry_Heroes
{
    partial class ManagerLogin
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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(ManagerLogin));
            lcaptcha = new Label();
            tbcans = new TextBox();
            lcpequal = new Label();
            lcp2 = new Label();
            lcpplus = new Label();
            lcp1 = new Label();
            bexit = new Button();
            bprevious = new Button();
            blogin = new Button();
            beyeok = new Button();
            beyecross = new Button();
            tbpass = new TextBox();
            tbusername = new TextBox();
            lpass = new Label();
            lusername = new Label();
            pictureBox1 = new PictureBox();
            brecapta = new Button();
            ((System.ComponentModel.ISupportInitialize)pictureBox1).BeginInit();
            SuspendLayout();
            // 
            // lcaptcha
            // 
            lcaptcha.Anchor = AnchorStyles.None;
            lcaptcha.AutoSize = true;
            lcaptcha.BackColor = Color.Transparent;
            lcaptcha.Font = new Font("Franklin Gothic Book", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            lcaptcha.ForeColor = Color.Gainsboro;
            lcaptcha.Image = (Image)resources.GetObject("lcaptcha.Image");
            lcaptcha.ImageAlign = ContentAlignment.MiddleLeft;
            lcaptcha.Location = new Point(260, 256);
            lcaptcha.Name = "lcaptcha";
            lcaptcha.Size = new Size(82, 20);
            lcaptcha.TabIndex = 63;
            lcaptcha.Text = "     Captcha";
            lcaptcha.TextAlign = ContentAlignment.MiddleRight;
            // 
            // tbcans
            // 
            tbcans.Anchor = AnchorStyles.None;
            tbcans.BackColor = SystemColors.ControlLight;
            tbcans.Cursor = Cursors.IBeam;
            tbcans.Font = new Font("Calibri", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            tbcans.Location = new Point(483, 254);
            tbcans.MaxLength = 3;
            tbcans.Name = "tbcans";
            tbcans.PlaceholderText = "?";
            tbcans.ScrollBars = ScrollBars.Horizontal;
            tbcans.Size = new Size(40, 26);
            tbcans.TabIndex = 62;
            // 
            // lcpequal
            // 
            lcpequal.Anchor = AnchorStyles.None;
            lcpequal.AutoSize = true;
            lcpequal.BackColor = Color.Black;
            lcpequal.BorderStyle = BorderStyle.Fixed3D;
            lcpequal.Font = new Font("Algerian", 14.25F, FontStyle.Regular, GraphicsUnit.Point);
            lcpequal.ForeColor = Color.WhiteSmoke;
            lcpequal.ImageAlign = ContentAlignment.MiddleLeft;
            lcpequal.Location = new Point(454, 256);
            lcpequal.Name = "lcpequal";
            lcpequal.Size = new Size(23, 23);
            lcpequal.TabIndex = 61;
            lcpequal.Text = "=";
            lcpequal.TextAlign = ContentAlignment.MiddleCenter;
            // 
            // lcp2
            // 
            lcp2.Anchor = AnchorStyles.None;
            lcp2.AutoSize = true;
            lcp2.BackColor = Color.Black;
            lcp2.BorderStyle = BorderStyle.Fixed3D;
            lcp2.Font = new Font("Algerian", 14.25F, FontStyle.Regular, GraphicsUnit.Point);
            lcp2.ForeColor = Color.WhiteSmoke;
            lcp2.ImageAlign = ContentAlignment.MiddleLeft;
            lcp2.Location = new Point(414, 256);
            lcp2.Name = "lcp2";
            lcp2.Size = new Size(2, 23);
            lcp2.TabIndex = 60;
            lcp2.TextAlign = ContentAlignment.MiddleCenter;
            // 
            // lcpplus
            // 
            lcpplus.Anchor = AnchorStyles.None;
            lcpplus.AutoSize = true;
            lcpplus.BackColor = Color.Black;
            lcpplus.BorderStyle = BorderStyle.Fixed3D;
            lcpplus.Font = new Font("Algerian", 14.25F, FontStyle.Regular, GraphicsUnit.Point);
            lcpplus.ForeColor = Color.WhiteSmoke;
            lcpplus.ImageAlign = ContentAlignment.MiddleLeft;
            lcpplus.Location = new Point(386, 256);
            lcpplus.Name = "lcpplus";
            lcpplus.Size = new Size(23, 23);
            lcpplus.TabIndex = 59;
            lcpplus.Text = "+";
            lcpplus.TextAlign = ContentAlignment.MiddleCenter;
            // 
            // lcp1
            // 
            lcp1.Anchor = AnchorStyles.None;
            lcp1.AutoSize = true;
            lcp1.BackColor = Color.Black;
            lcp1.BorderStyle = BorderStyle.Fixed3D;
            lcp1.Font = new Font("Algerian", 14.25F, FontStyle.Regular, GraphicsUnit.Point);
            lcp1.ForeColor = Color.WhiteSmoke;
            lcp1.ImageAlign = ContentAlignment.MiddleLeft;
            lcp1.Location = new Point(348, 256);
            lcp1.Name = "lcp1";
            lcp1.Size = new Size(2, 23);
            lcp1.TabIndex = 58;
            lcp1.TextAlign = ContentAlignment.MiddleCenter;
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
            bexit.Location = new Point(308, 367);
            bexit.Name = "bexit";
            bexit.Size = new Size(245, 34);
            bexit.TabIndex = 57;
            bexit.Text = "   Exit";
            bexit.UseVisualStyleBackColor = false;
            bexit.Click += bexit_Click;
            // 
            // bprevious
            // 
            bprevious.Anchor = AnchorStyles.None;
            bprevious.BackColor = Color.DarkGray;
            bprevious.Cursor = Cursors.Hand;
            bprevious.FlatAppearance.BorderColor = Color.FromArgb(192, 0, 0);
            bprevious.FlatAppearance.MouseDownBackColor = Color.DarkRed;
            bprevious.FlatAppearance.MouseOverBackColor = Color.Tomato;
            bprevious.FlatStyle = FlatStyle.Flat;
            bprevious.Font = new Font("Arial Rounded MT Bold", 9F, FontStyle.Regular, GraphicsUnit.Point);
            bprevious.Image = (Image)resources.GetObject("bprevious.Image");
            bprevious.ImageAlign = ContentAlignment.MiddleLeft;
            bprevious.Location = new Point(435, 326);
            bprevious.Name = "bprevious";
            bprevious.Size = new Size(119, 34);
            bprevious.TabIndex = 56;
            bprevious.Text = "   Previous";
            bprevious.UseVisualStyleBackColor = false;
            bprevious.Click += bprevious_Click;
            // 
            // blogin
            // 
            blogin.Anchor = AnchorStyles.None;
            blogin.BackColor = Color.DarkGray;
            blogin.BackgroundImageLayout = ImageLayout.Stretch;
            blogin.Cursor = Cursors.Hand;
            blogin.FlatAppearance.BorderColor = Color.FromArgb(192, 0, 0);
            blogin.FlatAppearance.MouseDownBackColor = Color.LightGreen;
            blogin.FlatAppearance.MouseOverBackColor = Color.Lime;
            blogin.FlatStyle = FlatStyle.Flat;
            blogin.Font = new Font("Arial Rounded MT Bold", 9F, FontStyle.Regular, GraphicsUnit.Point);
            blogin.Image = Properties.Resources.login;
            blogin.ImageAlign = ContentAlignment.MiddleLeft;
            blogin.Location = new Point(308, 326);
            blogin.Name = "blogin";
            blogin.Size = new Size(119, 34);
            blogin.TabIndex = 55;
            blogin.Text = "   Login";
            blogin.UseVisualStyleBackColor = false;
            blogin.Click += blogin_Click;
            // 
            // beyeok
            // 
            beyeok.Anchor = AnchorStyles.None;
            beyeok.BackColor = SystemColors.ScrollBar;
            beyeok.BackgroundImageLayout = ImageLayout.Stretch;
            beyeok.Cursor = Cursors.Hand;
            beyeok.FlatStyle = FlatStyle.Popup;
            beyeok.Image = (Image)resources.GetObject("beyeok.Image");
            beyeok.Location = new Point(523, 214);
            beyeok.Name = "beyeok";
            beyeok.Size = new Size(25, 23);
            beyeok.TabIndex = 54;
            beyeok.UseMnemonic = false;
            beyeok.UseVisualStyleBackColor = false;
            beyeok.Click += beyeok_Click;
            // 
            // beyecross
            // 
            beyecross.Anchor = AnchorStyles.None;
            beyecross.BackColor = SystemColors.ScrollBar;
            beyecross.BackgroundImageLayout = ImageLayout.Stretch;
            beyecross.Cursor = Cursors.Hand;
            beyecross.FlatStyle = FlatStyle.Popup;
            beyecross.Image = Properties.Resources.eyecross;
            beyecross.Location = new Point(523, 214);
            beyecross.Name = "beyecross";
            beyecross.Size = new Size(25, 23);
            beyecross.TabIndex = 53;
            beyecross.UseMnemonic = false;
            beyecross.UseVisualStyleBackColor = false;
            beyecross.Visible = false;
            beyecross.Click += beyecross_Click;
            // 
            // tbpass
            // 
            tbpass.Anchor = AnchorStyles.None;
            tbpass.BackColor = SystemColors.ScrollBar;
            tbpass.Cursor = Cursors.IBeam;
            tbpass.Font = new Font("Calibri", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            tbpass.Location = new Point(349, 214);
            tbpass.MaxLength = 25;
            tbpass.Name = "tbpass";
            tbpass.PlaceholderText = "Your Password";
            tbpass.ScrollBars = ScrollBars.Horizontal;
            tbpass.Size = new Size(174, 23);
            tbpass.TabIndex = 52;
            tbpass.UseSystemPasswordChar = true;
            // 
            // tbusername
            // 
            tbusername.Anchor = AnchorStyles.None;
            tbusername.BackColor = SystemColors.ScrollBar;
            tbusername.Cursor = Cursors.IBeam;
            tbusername.Font = new Font("Calibri", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            tbusername.Location = new Point(349, 177);
            tbusername.MaxLength = 25;
            tbusername.Name = "tbusername";
            tbusername.PlaceholderText = "Your Username";
            tbusername.ScrollBars = ScrollBars.Horizontal;
            tbusername.Size = new Size(199, 23);
            tbusername.TabIndex = 51;
            // 
            // lpass
            // 
            lpass.Anchor = AnchorStyles.None;
            lpass.AutoSize = true;
            lpass.BackColor = Color.Transparent;
            lpass.Font = new Font("Franklin Gothic Book", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            lpass.ForeColor = Color.Gainsboro;
            lpass.Image = (Image)resources.GetObject("lpass.Image");
            lpass.ImageAlign = ContentAlignment.MiddleLeft;
            lpass.Location = new Point(253, 215);
            lpass.Name = "lpass";
            lpass.Size = new Size(90, 20);
            lpass.TabIndex = 50;
            lpass.Text = "     Password";
            lpass.TextAlign = ContentAlignment.MiddleRight;
            // 
            // lusername
            // 
            lusername.Anchor = AnchorStyles.None;
            lusername.AutoSize = true;
            lusername.BackColor = Color.Transparent;
            lusername.Font = new Font("Franklin Gothic Book", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            lusername.ForeColor = Color.Gainsboro;
            lusername.Image = Properties.Resources.username1;
            lusername.ImageAlign = ContentAlignment.MiddleLeft;
            lusername.Location = new Point(245, 177);
            lusername.Name = "lusername";
            lusername.Size = new Size(98, 20);
            lusername.TabIndex = 49;
            lusername.Text = "      Username";
            lusername.TextAlign = ContentAlignment.MiddleRight;
            // 
            // pictureBox1
            // 
            pictureBox1.Anchor = AnchorStyles.None;
            pictureBox1.BackColor = Color.Transparent;
            pictureBox1.BackgroundImageLayout = ImageLayout.Stretch;
            pictureBox1.Image = Properties.Resources.Admin_Profile;
            pictureBox1.Location = new Point(348, 15);
            pictureBox1.Name = "pictureBox1";
            pictureBox1.Size = new Size(159, 159);
            pictureBox1.SizeMode = PictureBoxSizeMode.StretchImage;
            pictureBox1.TabIndex = 64;
            pictureBox1.TabStop = false;
            // 
            // brecapta
            // 
            brecapta.Anchor = AnchorStyles.None;
            brecapta.BackColor = SystemColors.ScrollBar;
            brecapta.BackgroundImageLayout = ImageLayout.Stretch;
            brecapta.Cursor = Cursors.Hand;
            brecapta.FlatStyle = FlatStyle.Popup;
            brecapta.Image = (Image)resources.GetObject("brecapta.Image");
            brecapta.Location = new Point(523, 254);
            brecapta.Name = "brecapta";
            brecapta.Size = new Size(25, 26);
            brecapta.TabIndex = 82;
            brecapta.UseMnemonic = false;
            brecapta.UseVisualStyleBackColor = false;
            brecapta.Click += brecapta_Click;
            // 
            // ManagerLogin
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            BackgroundImage = Properties.Resources.background_gda7ff7ad6_1280;
            BackgroundImageLayout = ImageLayout.Stretch;
            ClientSize = new Size(784, 511);
            Controls.Add(brecapta);
            Controls.Add(pictureBox1);
            Controls.Add(lcaptcha);
            Controls.Add(tbcans);
            Controls.Add(lcpequal);
            Controls.Add(lcp2);
            Controls.Add(lcpplus);
            Controls.Add(lcp1);
            Controls.Add(bexit);
            Controls.Add(bprevious);
            Controls.Add(blogin);
            Controls.Add(beyeok);
            Controls.Add(beyecross);
            Controls.Add(tbpass);
            Controls.Add(tbusername);
            Controls.Add(lpass);
            Controls.Add(lusername);
            Icon = (Icon)resources.GetObject("$this.Icon");
            Name = "ManagerLogin";
            StartPosition = FormStartPosition.CenterScreen;
            Text = "Manager Login";
            ((System.ComponentModel.ISupportInitialize)pictureBox1).EndInit();
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private Label lcaptcha;
        private TextBox tbcans;
        private Label lcpequal;
        private Label lcp2;
        private Label lcpplus;
        private Label lcp1;
        private Button bexit;
        private Button bprevious;
        private Button blogin;
        private Button beyeok;
        private Button beyecross;
        private TextBox tbpass;
        private TextBox tbusername;
        private Label lpass;
        private Label lusername;
        private PictureBox pictureBox1;
        private Button brecapta;
    }
}