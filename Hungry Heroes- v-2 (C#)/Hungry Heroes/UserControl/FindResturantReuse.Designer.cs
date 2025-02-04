namespace Hungry_Heroes
{
    partial class FindResturantReuse
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

        #region Component Designer generated code

        /// <summary> 
        /// Required method for Designer support - do not modify 
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(FindResturantReuse));
            lresname = new Label();
            bshowmenu = new Button();
            display = new DataGridView();
            tbsearch = new TextBox();
            button1 = new Button();
            bsearch = new Button();
            ((System.ComponentModel.ISupportInitialize)display).BeginInit();
            SuspendLayout();
            // 
            // lresname
            // 
            lresname.AutoSize = true;
            lresname.BackColor = Color.Transparent;
            lresname.Font = new Font("Arial Rounded MT Bold", 14.25F, FontStyle.Underline, GraphicsUnit.Point);
            lresname.Location = new Point(258, 24);
            lresname.Name = "lresname";
            lresname.Size = new Size(199, 22);
            lresname.TabIndex = 0;
            lresname.Text = "Available Resturants";
            // 
            // bshowmenu
            // 
            bshowmenu.Anchor = AnchorStyles.Left;
            bshowmenu.BackColor = Color.Transparent;
            bshowmenu.Cursor = Cursors.Hand;
            bshowmenu.FlatAppearance.BorderSize = 3;
            bshowmenu.FlatAppearance.MouseDownBackColor = Color.Gray;
            bshowmenu.FlatAppearance.MouseOverBackColor = Color.FromArgb(64, 64, 64);
            bshowmenu.FlatStyle = FlatStyle.Popup;
            bshowmenu.Font = new Font("Arial Rounded MT Bold", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            bshowmenu.ForeColor = Color.Black;
            bshowmenu.Image = (Image)resources.GetObject("bshowmenu.Image");
            bshowmenu.ImageAlign = ContentAlignment.MiddleLeft;
            bshowmenu.Location = new Point(468, 404);
            bshowmenu.Name = "bshowmenu";
            bshowmenu.Size = new Size(186, 33);
            bshowmenu.TabIndex = 14;
            bshowmenu.Text = "Show Menu";
            bshowmenu.UseVisualStyleBackColor = false;
            bshowmenu.Click += bshowmenu_Click;
            // 
            // display
            // 
            display.AllowUserToAddRows = false;
            display.AllowUserToDeleteRows = false;
            display.AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.Fill;
            display.BackgroundColor = SystemColors.ActiveBorder;
            display.BorderStyle = BorderStyle.Fixed3D;
            display.ColumnHeadersHeightSizeMode = DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            display.Location = new Point(87, 77);
            display.MultiSelect = false;
            display.Name = "display";
            display.ReadOnly = true;
            display.RowTemplate.Height = 25;
            display.Size = new Size(567, 321);
            display.TabIndex = 15;
            display.CellClick += display_CellClick;
            // 
            // tbsearch
            // 
            tbsearch.BorderStyle = BorderStyle.None;
            tbsearch.Font = new Font("Segoe UI", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            tbsearch.Location = new Point(87, 58);
            tbsearch.Name = "tbsearch";
            tbsearch.PlaceholderText = "Enter Resturant Name for Searching Resturant ";
            tbsearch.Size = new Size(567, 18);
            tbsearch.TabIndex = 16;
            // 
            // button1
            // 
            button1.Anchor = AnchorStyles.Left;
            button1.BackColor = Color.Transparent;
            button1.Cursor = Cursors.Hand;
            button1.FlatAppearance.BorderSize = 3;
            button1.FlatAppearance.MouseDownBackColor = Color.Gray;
            button1.FlatAppearance.MouseOverBackColor = Color.FromArgb(64, 64, 64);
            button1.FlatStyle = FlatStyle.Popup;
            button1.Font = new Font("Arial Rounded MT Bold", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            button1.ForeColor = Color.Black;
            button1.Image = (Image)resources.GetObject("button1.Image");
            button1.ImageAlign = ContentAlignment.MiddleLeft;
            button1.Location = new Point(279, 404);
            button1.Name = "button1";
            button1.Size = new Size(186, 33);
            button1.TabIndex = 17;
            button1.Text = "   Search Resturant";
            button1.UseVisualStyleBackColor = false;
            button1.Click += button1_Click;
            // 
            // bsearch
            // 
            bsearch.Anchor = AnchorStyles.Left;
            bsearch.BackColor = Color.Transparent;
            bsearch.Cursor = Cursors.Hand;
            bsearch.FlatAppearance.BorderSize = 3;
            bsearch.FlatAppearance.MouseDownBackColor = Color.Gray;
            bsearch.FlatAppearance.MouseOverBackColor = Color.FromArgb(64, 64, 64);
            bsearch.FlatStyle = FlatStyle.Popup;
            bsearch.Font = new Font("Arial Rounded MT Bold", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            bsearch.ForeColor = Color.Black;
            bsearch.Image = (Image)resources.GetObject("bsearch.Image");
            bsearch.ImageAlign = ContentAlignment.MiddleLeft;
            bsearch.Location = new Point(87, 404);
            bsearch.Name = "bsearch";
            bsearch.Size = new Size(186, 33);
            bsearch.TabIndex = 18;
            bsearch.Text = "   Refresh";
            bsearch.UseVisualStyleBackColor = false;
            bsearch.Click += bsearch_Click;
            // 
            // FindResturantReuse
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            BackColor = Color.Silver;
            Controls.Add(bsearch);
            Controls.Add(button1);
            Controls.Add(tbsearch);
            Controls.Add(display);
            Controls.Add(bshowmenu);
            Controls.Add(lresname);
            Name = "FindResturantReuse";
            Size = new Size(745, 476);
            Load += FindResturantReuse_Load;
            ((System.ComponentModel.ISupportInitialize)display).EndInit();
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private Label lresname;
        private Button bshowmenu;
        private DataGridView display;
        private TextBox tbsearch;
        private Button button1;
        private Button bsearch;
    }
}
